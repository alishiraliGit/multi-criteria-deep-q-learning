from collections import OrderedDict
import pickle
import sys
import time
import numpy as np
from scipy.stats import spearmanr
import torch

from rlcodebase.eval import metrics
from rlcodebase.infrastructure.utils import rl_utils
from rlcodebase.infrastructure.utils import pytorch_utils as ptu
from rlcodebase.infrastructure.logger import Logger
from rlcodebase.agents.dqn_agent import DQNAgent
from rlcodebase.envs.gym_utils import init_gym_and_update_params
from rlcodebase.envs.mimic_utils import init_mimic_and_update_params
from rlcodebase.pruners.dqn_pruner import MDQNPruner, ExtendedMDQNPruner


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
        self.log_metrics = None
        self.log_params = None
        self.mean_episode_reward = -np.nan
        self.best_mean_episode_reward = -np.inf
        self.best_performance = -np.inf

        #############
        # ENV
        #############
        self.offline = params['offline']
        self.params['agent_params']['offline'] = params['offline']
        self.bcq = params['bcq']

        if not self.offline:
            self.env = init_gym_and_update_params(params)
        else:
            self.env, self.train_paths, self.test_paths = init_mimic_and_update_params(params)

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
                print('\n\n********** Iteration %i ************' % itr)

            # Decide if metrics should be logged
            if itr % self.params['scalar_log_freq'] == 0 and self.params['scalar_log_freq'] != -1:
                self.log_metrics = True
            else:
                self.log_metrics = False

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
                    train_paths, envsteps_this_batch = None, 1
                else:
                    use_batch_size = self.params['batch_size']
                    if itr == 0:
                        use_batch_size = self.params['batch_size_initial']
                    train_paths, envsteps_this_batch = (
                        self.collect_training_trajectories(
                            itr, initial_expertdata, collect_policy, use_batch_size)
                    )
                test_paths = None
            else:
                train_paths, test_paths = self.train_paths, self.test_paths

                envsteps_this_batch = sum([len(path['reward']) for path in train_paths])

            if not self.offline:
                self.total_envsteps += envsteps_this_batch
            else:
                self.total_envsteps = envsteps_this_batch

            # Relabel the collected obs with actions from a provided expert policy
            if relabel_with_expert and itr >= start_relabel_with_expert:
                train_paths = self.do_relabel_with_expert(expert_policy, train_paths)

            # Add collected data to replay buffer
            if self.offline and itr > 0:
                pass
            else:
                self.agent.add_to_replay_buffer(train_paths)

            # Train agent (using sampled data from replay buffer)
            if itr % print_period == 0:
                print('\nTraining agent...')

            all_logs = self.train_agent()

            # Log/save
            performance = -np.inf
            if self.log_metrics:
                # Perform logging
                print('\nBeginning logging procedure (%s)...' % self.params['logdir'])
                if isinstance(self.agent, DQNAgent):
                    if not self.offline:
                        performance = self.perform_dqn_logging(all_logs)
                    else:
                        if self.agent.mdqn or self.agent.emdqn:
                            performance = self.perform_mdqn_offline_logging(itr, train_paths, test_paths, all_logs)
                        else:
                            performance = self.perform_dqn_offline_logging(itr, train_paths, test_paths, all_logs)
                else:
                    self.perform_logging(itr, train_paths, eval_policy, all_logs)

            if self.log_params:
                if isinstance(self.agent, DQNAgent):
                    if (not self.params['save_best']) \
                            or (self.params['save_best'] and (performance >= self.best_performance)):
                        save_path = '{}/dqn_agent.pt'.format(self.params['logdir'])
                        self.agent.critic.save(save_path)
                        print('agent saved!')

                        if performance >= self.best_performance:
                            self.best_performance = performance
                else:
                    raise NotImplementedError

    @staticmethod
    def do_relabel_with_expert(expert_policy, paths):
        print('\nRelabelling collected observations with labels from an expert policy...')

        # Relabel collected observations (from our policy) with labels from an expert policy
        for path in paths:
            path['action'] = expert_policy.get_action(path['observation'])

        return paths

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
        paths, envsteps_this_batch = rl_utils.sample_trajectories(
            self.env, collect_policy,
            min_timesteps_per_batch=num_transitions_to_sample,
            max_path_length=self.params['ep_len'],
            render=False
        )

        return paths, envsteps_this_batch

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

    def perform_dqn_logging(self, all_logs):
        last_log = all_logs[-1]

        if self.params['env_name'] != 'LunarLander-MultiInterRewardNoise' and \
                (self.params['mdqn'] or self.params['emdqn']):
            episode_rewards = self.env.episode_final_rewards
        else:
            episode_rewards = self.env.get_episode_rewards()

        if len(episode_rewards) > 0:
            self.mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            self.best_mean_episode_reward = np.maximum(self.best_mean_episode_reward, self.mean_episode_reward)

        logs = OrderedDict()

        logs['Train_EnvstepsSoFar'] = self.agent.t
        print('Timestep %d' % (self.agent.t,))
        if self.mean_episode_reward > -5000:
            logs['Train_AverageReturn'] = np.mean(self.mean_episode_reward)
        print('mean reward (100 episodes) %f' % self.mean_episode_reward)
        if self.best_mean_episode_reward > -5000:
            logs['Train_BestReturn'] = np.mean(self.best_mean_episode_reward)
        print('best mean reward %f' % self.best_mean_episode_reward)

        if self.start_time is not None:
            time_since_start = (time.time() - self.start_time)
            print('running time %f' % time_since_start)
            logs['TimeSinceStart'] = time_since_start

        logs.update(last_log)

        sys.stdout.flush()

        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, self.agent.t)
        print('Done logging...\n\n')

        self.logger.flush()

        return logs.get('Train_AverageReturn', -np.inf)

    def perform_logging(self, itr, train_paths, eval_policy, all_logs):

        last_log = all_logs[-1]

        # Collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = \
            rl_utils.sample_trajectories(self.env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'])

        # save eval metrics
        if self.log_metrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in train_paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in train_paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs['Eval_AverageReturn'] = np.mean(eval_returns)
            logs['Eval_StdReturn'] = np.std(eval_returns)
            logs['Eval_MaxReturn'] = np.max(eval_returns)
            logs['Eval_MinReturn'] = np.min(eval_returns)
            logs['Eval_AverageEpLen'] = np.mean(eval_ep_lens)

            logs['Train_AverageReturn'] = np.mean(train_returns)
            logs['Train_StdReturn'] = np.std(train_returns)
            logs['Train_MaxReturn'] = np.max(train_returns)
            logs['Train_MinReturn'] = np.min(train_returns)
            logs['Train_AverageEpLen'] = np.mean(train_ep_lens)

            logs['Train_EnvstepsSoFar'] = self.total_envsteps
            logs['TimeSinceStart'] = time.time() - self.start_time
            logs.update(last_log)

            # Changed by Ali
            if itr == 0:
                initial_return = np.mean(train_returns)
                logs['Initial_DataCollection_AverageReturn'] = initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()

    def perform_dqn_offline_logging(self, itr, _train_paths, eval_paths, all_logs):

        last_log = all_logs[-1]

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
            if self.bcq:
                qa_values_na = ptu.to_numpy(self.agent.critic.qa_values(obs_n)[0])
            else:    
                qa_values_na = self.agent.critic.qa_values(obs_n)

            qa_values_na = ptu.from_numpy(qa_values_na)
            ac_n = ptu.from_numpy(ac_n).to(torch.long)

            q_values_n = torch.gather(qa_values_na, 1, ac_n.unsqueeze(1)).squeeze(1)

            q_values_n = ptu.to_numpy(q_values_n)

            # Get reward-to-go
            rtg_n = rl_utils.discounted_cumsum(re_n, self.params['gamma'])
            rtg_mort_n = rl_utils.discounted_cumsum(re_mort_n, self.params['gamma'])

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

        t_test = metrics.TTest(th=0)
        t_test_res = t_test.value(None, None, None, None, None, all_rtgs_mort, all_q_values, None)

        diff_survival_quantile_test = metrics.DiffSurvivalQuantiles(q=0.25, th=0)
        diff_survival_quantile_test_res = diff_survival_quantile_test.value(
            None, None, None, None, None, all_rtgs_mort, all_q_values, None
        )

        avg_q = np.mean(all_q_values)

        # save eval metrics
        if self.log_metrics:
            # decide what to log
            logs = OrderedDict()
            logs['Rho'] = rho
            logs['Rho_mort'] = rho_mort
            logs['Avg Q'] = avg_q
            logs['T_stat'] = t_test_res['statistic']
            logs['T_test_p'] = t_test_res['pvalue']
            logs['Diff_Survival_Quantile_mean'] = diff_survival_quantile_test_res['mean']
            logs['Diff_Survival_Quantile_std'] = diff_survival_quantile_test_res['std']

            logs['TimeSinceStart'] = time.time() - self.start_time
            logs['Train_itr'] = itr
            logs.update(last_log)

            # Perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()

            return diff_survival_quantile_test_res['mean']  # TODO: Update to t_stat and repeat experiments

    def perform_mdqn_offline_logging(self, itr, _train_paths, eval_paths, all_logs):

        last_log = all_logs[-1]

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
            rtg_mort_n = rl_utils.discounted_cumsum(re_mort_n, self.params['gamma'])

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
