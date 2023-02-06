from collections import OrderedDict
import pickle
import os
import sys
import time
import gym
from gym import wrappers
import numpy as np
from scipy.stats import spearmanr
import torch
from sklearn.model_selection import train_test_split

from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure.atari_wrappers import ReturnWrapper
from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.agents.dqn_agent import DQNAgent
from cs285.infrastructure.dqn_utils import register_custom_envs
from cs285.pruners.dqn_pruner import MDQNPruner, ExtendedMDQNPruner

# How many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below


class RLTrainer(object):

    def __init__(self, params):

        #############
        # INIT
        #############

        # Get params, create logger
        self.params = params
        self.logger = Logger(self.params['logdir'])

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)

        # To be set later in run_training_loop
        self.total_envsteps = None
        self.start_time = None
        self.log_video = None
        self.log_metrics = None
        self.log_params = None
        self.best_performance = 0

        #############
        # ENV
        #############
        self.offline = params['offline']
        self.params['agent_params']['offline'] = params['offline']

        # Import plotting (locally if 'obstacles' env)
        if not (self.params['env_name'] == 'obstacles-cs285-v0'):
            import matplotlib
            matplotlib.use('Agg')

        #####################
        # Online learning
        #####################
        if not self.offline:
            # Make the gym environment
            register_custom_envs()

            self.env = gym.make(self.params['env_name'])

            # Added by Ali
            if params['env_name'] == 'LunarLander-Customizable' and params['env_rew_weights'] is not None:
                self.env.set_rew_weights(params['env_rew_weights'])

            if self.params['video_log_freq'] > 0:
                self.episode_trigger = lambda episode: episode % self.params['video_log_freq'] == 0
            else:
                self.episode_trigger = lambda episode: False

            if 'env_wrappers' in self.params:
                # These operations are currently only for Atari envs
                self.env = wrappers.RecordEpisodeStatistics(self.env, deque_size=1000)
                self.env = ReturnWrapper(self.env)
                self.env = wrappers.RecordVideo(self.env, os.path.join(self.params['logdir'], 'gym'),
                                                episode_trigger=self.episode_trigger)
                self.env = params['env_wrappers'](self.env)
                self.mean_episode_reward = -float('nan')
                self.best_mean_episode_reward = -float('inf')

            if 'non_atari_colab_env' in self.params and self.params['video_log_freq'] > 0:
                self.env = wrappers.RecordVideo(self.env, os.path.join(self.params['logdir'], 'gym'),
                                                episode_trigger=self.episode_trigger)
                self.mean_episode_reward = -float('nan')
                self.best_mean_episode_reward = -float('inf')

            self.env.seed(seed)

            # Maximum length for episodes
            self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
            global MAX_VIDEO_LEN
            MAX_VIDEO_LEN = self.params['ep_len']

            # Is this env continuous, or discrete?
            discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
            self.params['agent_params']['discrete'] = discrete

            # Are the observations images?
            img = len(self.env.observation_space.shape) > 2

            # Observation and action sizes
            ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
            ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
            re_dim = self.env.reward_dim
            self.params['agent_params']['ac_dim'] = ac_dim
            self.params['agent_params']['ob_dim'] = ob_dim
            self.params['agent_params']['re_dim'] = re_dim

            self.params['agent_params']['cql'] = self.params['cql']

            # Simulation timestep, will be used for video saving
            if 'model' in dir(self.env):
                self.fps = 1 / self.env.model.opt.timestep
            elif 'env_wrappers' in self.params:
                self.fps = 30  # this is not actually used when using the Monitor wrapper
            elif 'video.frames_per_second' in self.env.env.metadata.keys():
                self.fps = self.env.env.metadata['video.frames_per_second']
            else:
                self.fps = 10

        #####################
        # Offline learning
        #####################
        else:
            self.env = 'MIMIC'

            # Load the data
            with open(self.params['buffer_path'], 'rb') as f:
                all_paths = pickle.load(f)

            if self.params['env_name'] == 'MIMIC':
                all_paths = utils.format_reward(all_paths, weights=params['env_rew_weights'])
            if self.params['env_name'] == 'MIMIC-Continuous':
                all_paths = utils.format_reward(all_paths, weights=params['env_rew_weights'], continuous=True)
            elif self.params['env_name'] == 'MIMIC-MultiInterReward':
                all_paths = utils.format_reward(all_paths, multi_inter=True)
            elif self.params['env_name'] == 'MIMIC-MultiReward':
                all_paths = utils.format_reward(all_paths, multi=True)
            elif self.params['env_name'] == 'MIMIC-MultiContinuousReward':
                all_paths = utils.format_reward(all_paths, multi_continuous=True)
            else:
                raise Exception('Invalid env_name!')

            # Let's use 5% as validation set and 15% as hold-out set
            self.paths, self.test_paths = train_test_split(all_paths, test_size=0.05, random_state=seed)

            # Is this env continuous, or discrete?
            discrete = True
            self.params['agent_params']['discrete'] = discrete

            # Are the observations images?
            img = False

            # Observation and action sizes
            ob_shape = self.paths[0]['observation'].shape
            ob_dim = 1 if len(ob_shape) == 1 else ob_shape[1]
            ac_dim = 25
            re_shape = self.paths[0]['reward'].shape
            re_dim = 1 if len(re_shape) == 1 else re_shape[1]
            self.params['agent_params']['ac_dim'] = ac_dim
            self.params['agent_params']['ob_dim'] = ob_dim
            self.params['agent_params']['re_dim'] = re_dim

            self.params['agent_params']['cql'] = self.params['cql']

        #############
        # AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                          initial_expertdata=None, relabel_with_expert=False,
                          start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert: whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # Init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        # TODO: Hard-coded
        print_period = 1000 if isinstance(self.agent, DQNAgent) else 1

        for itr in range(n_iter):
            if itr % print_period == 0:
                print("\n\n********** Iteration %i ************" % itr)

            # Decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.log_video = True
            else:
                self.log_video = False

            # Decide if metrics should be logged
            if itr % self.params['scalar_log_freq'] == 0 and self.params['scalar_log_freq'] != -1:
                self.log_metrics = True
            else:
                self.log_metrics = False

            # Added by Ali
            # Decide if network parameters should be logged
            if itr % self.params['params_log_freq'] == 0 and self.params['params_log_freq'] != -1:
                self.log_params = True
            else:
                self.log_params = False

            # Collect trajectories, to be used for training
            # TODO: Fix: For single step trajectories, agent should be DQNAgent.
            if not self.offline:
                if isinstance(self.agent, DQNAgent):
                    # Only perform an env step and add to replay buffer for DQN
                    self.agent.step_env()
                    # The step_env() automatically adds to replay buffer so paths are not required
                    paths, envsteps_this_batch, train_video_paths = None, 1, None
                else:
                    use_batch_size = self.params['batch_size']
                    if itr == 0:
                        use_batch_size = self.params['batch_size_initial']
                    paths, envsteps_this_batch, train_video_paths = (
                        self.collect_training_trajectories(
                            itr, initial_expertdata, collect_policy, use_batch_size)
                    )
                test_paths = None
            else:
                paths, test_paths = self.paths, self.test_paths

                envsteps_this_batch = sum([len(path['reward']) for path in paths])
                train_video_paths = None

            if not self.offline:
                self.total_envsteps += envsteps_this_batch
            else:
                self.total_envsteps = envsteps_this_batch

            # Relabel the collected obs with actions from a provided expert policy
            if relabel_with_expert and itr >= start_relabel_with_expert:
                paths = self.do_relabel_with_expert(expert_policy, paths)

            # Add collected data to replay buffer
            if self.offline and itr > 0:
                pass
            else:
                self.agent.add_to_replay_buffer(paths)

            # Train agent (using sampled data from replay buffer)
            if itr % print_period == 0:
                print("\nTraining agent...")

            all_logs = self.train_agent()

            # Log/save
            performance = 0
            if self.log_video or self.log_metrics:
                # Perform logging
                print('\nBeginning logging procedure (%s)...' % self.params['logdir'])
                if isinstance(self.agent, DQNAgent):
                    if not self.offline:
                        self.perform_dqn_logging(all_logs)
                    else:
                        if self.agent.mdqn or self.agent.emdqn:
                            performance = self.perform_mdqn_offline_logging(itr, paths, test_paths, all_logs)
                        else:
                            performance = self.perform_dqn_offline_logging(itr, paths, test_paths, all_logs)
                else:
                    self.perform_logging(itr, paths, eval_policy, train_video_paths, all_logs)

            if self.log_params:
                if isinstance(self.agent, DQNAgent):
                    if (not self.params['save_best']) \
                            or (self.params['save_best'] and (performance >= self.best_performance)):
                        save_path = '{}/dqn_agent.pt'.format(self.params['logdir'])
                        self.agent.critic.save(save_path)

                        if performance >= self.best_performance:
                            self.best_performance = performance
                else:
                    raise NotImplementedError

    ####################################
    ####################################

    @staticmethod
    def do_relabel_with_expert(expert_policy, paths):
        print('\nRelabelling collected observations with labels from an expert policy...')

        # Relabel collected observations (from our policy) with labels from an expert policy
        for path in paths:
            path['action'] = expert_policy.get_action(path['observation'])

        return paths

    ####################################
    ####################################

    def collect_training_trajectories(self, itr, initial_expertdata, collect_policy, num_transitions_to_sample):
        """
        :param itr:
        :param initial_expertdata: path to expert data pkl file
        :param collect_policy: the current policy using which we collect data
        :param num_transitions_to_sample: the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """
        # Decide whether to load training data or use the current policy to collect more data
        if (itr == 0) and (initial_expertdata is not None):
            with open(initial_expertdata, 'rb') as f:
                loaded_paths = pickle.load(f)
            return loaded_paths, 0, None

        # Collect 'num_transitions_to_sample' samples to be used for training
        # HINT1: use sample_trajectories from utils
        # HINT2: you want each of these collected rollouts to be of length self.params['ep_len']
        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = utils.sample_trajectories(
            self.env,
            collect_policy,
            min_timesteps_per_batch=num_transitions_to_sample,
            max_path_length=self.params['ep_len'],
            render=False
        )

        # Collect more rollouts with the same policy, to be saved as videos in tensorboard
        # Note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN
        train_video_paths = None
        if self.log_video:
            print('\nCollecting train rollouts to be used for saving videos...')
            train_video_paths = \
                utils.sample_n_trajectories(self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, render=True)

        return paths, envsteps_this_batch, train_video_paths

    ####################################
    ####################################

    def train_agent(self):
        # print('\nTraining agent using sampled data from replay buffer...')
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            # Sample some data from the data buffer
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = \
                self.agent.sample(self.params['batch_size'])

            # Use the sampled data to train an agent
            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)

            all_logs.append(train_log)

        return all_logs

    ####################################
    ####################################

    def perform_dqn_logging(self, all_logs):
        last_log = all_logs[-1]

        episode_rewards = self.env.get_episode_rewards()

        if len(episode_rewards) > 0:
            self.mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            self.best_mean_episode_reward = np.maximum(self.best_mean_episode_reward, self.mean_episode_reward)

        logs = OrderedDict()

        logs["Train_EnvstepsSoFar"] = self.agent.t
        print("Timestep %d" % (self.agent.t,))
        if self.mean_episode_reward > -5000:
            logs["Train_AverageReturn"] = np.mean(self.mean_episode_reward)
        print("mean reward (100 episodes) %f" % self.mean_episode_reward)
        if self.best_mean_episode_reward > -5000:
            logs["Train_BestReturn"] = np.mean(self.best_mean_episode_reward)
        print("best mean reward %f" % self.best_mean_episode_reward)

        if self.start_time is not None:
            time_since_start = (time.time() - self.start_time)
            print("running time %f" % time_since_start)
            logs["TimeSinceStart"] = time_since_start

        logs.update(last_log)

        sys.stdout.flush()

        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, self.agent.t)
        print('Done logging...\n\n')

        self.logger.flush()

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, all_logs):

        last_log = all_logs[-1]

        #######################

        # Collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = \
            utils.sample_trajectories(self.env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'])

        # Save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths is not None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            # Save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='eval_rollouts')

        #######################

        # save eval metrics
        if self.log_metrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            # Changed by Ali
            if itr == 0:
                initial_return = np.mean(train_returns)
                logs["Initial_DataCollection_AverageReturn"] = initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()

    def perform_dqn_offline_logging(self, itr, train_paths, eval_paths, all_logs):

        last_log = all_logs[-1]

        # if len(last_log) == 0:
        #    return 0

        # Run for test set
        all_q_values = []
        all_rtgs = []
        all_rtgs_mort = []
        for eval_path in eval_paths:
            obs_n = eval_path['observation']
            if obs_n.ndim == 1:
                obs_n = obs_n[:, np.newaxis]

            ac_n = eval_path['action']
            re_n = eval_path['reward']
            re_mort_n = eval_path['sparse_90d_rew']

            # Get the Q-values
            qa_values_na = self.agent.critic.qa_values(obs_n)

            qa_values_na = ptu.from_numpy(qa_values_na)
            ac_n = ptu.from_numpy(ac_n).to(torch.long)

            q_values_n = torch.gather(qa_values_na, 1, ac_n.unsqueeze(1)).squeeze(1)

            q_values_n = ptu.to_numpy(q_values_n)

            # Get reward-to-go
            rtg_n = utils.discounted_cumsum(re_n, self.params['gamma'])
            rtg_mort_n = utils.discounted_cumsum(re_mort_n, self.params['gamma'])

            # Append
            all_q_values.append(q_values_n)
            all_rtgs.append(rtg_n)
            all_rtgs_mort.append(rtg_mort_n)

        all_q_values = np.concatenate(all_q_values, axis=0)
        all_rtgs = np.concatenate(all_rtgs, axis=0)
        all_rtgs_mort = np.concatenate(all_rtgs_mort, axis=0)

        # noinspection PyTypeChecker, PyUnresolvedReferences
        rho = spearmanr(all_rtgs, all_q_values).correlation
        # noinspection PyTypeChecker, PyUnresolvedReferences
        rho_mort = spearmanr(all_rtgs_mort, all_q_values).correlation

        avg_q = np.mean(all_q_values)

        # Run for train set

        # all_q_values = []
        # all_rtgs = []
        # all_rtgs_mort = []
        # for train_path in train_paths:
        #     obs_n = train_path['observation']
        #     if obs_n.ndim == 1:
        #         obs_n = obs_n[:, np.newaxis]
        #
        #     ac_n = train_path['action']
        #     re_n = train_path['reward']
        #     re_mort_n = train_path['sparse_90d_rew']
        #
        #     # Get the Q-values
        #     qa_values_na = self.agent.critic.qa_values(obs_n)
        #
        #     qa_values_na = ptu.from_numpy(qa_values_na)
        #     ac_n = ptu.from_numpy(ac_n).to(torch.long)
        #
        #     q_values_n = torch.gather(qa_values_na, 1, ac_n.unsqueeze(1)).squeeze(1)
        #
        #     q_values_n = ptu.to_numpy(q_values_n)
        #
        #     # Get reward-to-go
        #     rtg_n = utils.discounted_cumsum(re_n, self.params['gamma'])
        #     rtg_mort_n = utils.discounted_cumsum(re_mort_n, self.params['gamma'])
        #
        #     # Append
        #     all_q_values.append(q_values_n)
        #     all_rtgs.append(rtg_n)
        #     all_rtgs_mort.append(rtg_mort_n)
        #
        # all_q_values = np.concatenate(all_q_values, axis=0)
        # all_rtgs = np.concatenate(all_rtgs, axis=0)
        # all_rtgs_mort = np.concatenate(all_rtgs_mort, axis=0)
        #
        # # noinspection PyTypeChecker, PyUnresolvedReferences
        # rho_train = spearmanr(all_rtgs, all_q_values).correlation
        # # noinspection PyTypeChecker, PyUnresolvedReferences
        # rho_train_mort = spearmanr(all_rtgs_mort, all_q_values).correlation
        #
        # avg_q_train = np.mean(all_q_values)

        # save eval metrics
        if self.log_metrics:
            # decide what to log
            logs = OrderedDict()
            logs['Rho'] = rho
            logs['Rho_mort'] = rho_mort
            logs['Avg Q'] = avg_q

            # logs['Rho_train'] = rho_train
            # logs['Rho_mort_train'] = rho_train_mort
            # logs['Avg Q train'] = avg_q_train
            logs['TimeSinceStart'] = time.time() - self.start_time
            logs['Train_itr'] = itr
            logs.update(last_log)

            # Perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()

            return rho_mort

    def perform_mdqn_offline_logging(self, itr, _train_paths, eval_paths, all_logs):

        last_log = all_logs[-1]

        # if len(last_log) == 0:
        #    return 0

        # Get the pruner
        if self.agent.mdqn:
            pruner = MDQNPruner(n_draw=self.params['pruning_n_draw'], critic=self.agent.critic)
        elif self.agent.emdqn:
            pruner = ExtendedMDQNPruner(n_draw=self.params['pruning_n_draw'], critic=self.agent.critic)
        else:
            raise NotImplementedError

        # Run on the test set
        tp = 0
        p = 0
        fn = 0
        all_num_available = []
        all_rtg_pruned = []
        all_rtg_available = []
        for eval_path in eval_paths:
            obs_n = eval_path['observation']
            if obs_n.ndim == 1:
                obs_n = obs_n[:, np.newaxis]

            # Check action is included
            ac_n = eval_path['action'].astype(int)
            available_acs_n = pruner.get_list_of_available_actions(obs_n)

            all_num_available.extend([len(available_acs_n[i_ac]) for i_ac, ac in enumerate(ac_n)])

            tp += np.sum([(ac in available_acs_n[i_ac]) for i_ac, ac in enumerate(ac_n)])
            p += np.sum([len(available_acs_n[i_ac]) for i_ac, ac in enumerate(ac_n)])
            fn += np.sum([(ac not in available_acs_n[i_ac]) for i_ac, ac in enumerate(ac_n)])

            # Compare pruned and remained actions based on rtg
            re_mort_n = eval_path['sparse_90d_rew']
            rtg_mort_n = utils.discounted_cumsum(re_mort_n, self.params['gamma'])

            for i_ac, ac in enumerate(ac_n):
                if ac in available_acs_n[i_ac]:
                    all_rtg_available.append(rtg_mort_n[i_ac])
                else:
                    all_rtg_pruned.append(rtg_mort_n[i_ac])

        # Decide what to log
        logs = OrderedDict()
        logs['Recall'] = tp/(tp + fn)
        logs['Precision'] = tp/p
        logs['F1'] = 2*logs['Recall']*logs['Precision'] / (logs['Recall'] + logs['Precision'])
        logs['Avg_Num_Available_Actions'] = np.mean(all_num_available)
        logs['RTG_Available_Actions'] = np.mean(all_rtg_available)
        logs['RTG_Pruned_Actions'] = np.mean(all_rtg_pruned)
        logs['RTG_Available_to_Pruned_Ratio'] = logs['RTG_Available_Actions'] / logs['RTG_Pruned_Actions']

        # Save eval metrics
        logs['TimeSinceStart'] = time.time() - self.start_time
        logs['Train_itr'] = itr
        logs.update(last_log)

        # Perform the logging
        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, itr)
        print('Done logging...\n\n')

        self.logger.flush()

        return logs['F1']
