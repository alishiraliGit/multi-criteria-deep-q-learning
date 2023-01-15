import pickle
import os

import numpy as np
import torch
import gym
from gym import wrappers
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from cs285.infrastructure.atari_wrappers import ReturnWrapper
from cs285.infrastructure import utils
from cs285.infrastructure.dqn_utils import register_custom_envs
from cs285.pruners.base_pruner import BasePruner
from cs285.infrastructure import pytorch_util as ptu


class RLEvaluator(object):

    def __init__(self, params):

        #############
        # INIT
        #############

        # Get params, create logger
        self.params = params
        self.log_dir = self.params['logdir']

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)

        # To be set later in run_training_loop
        self.total_envsteps = None
        self.start_time = None

        #############
        # ENV
        #############
        if not self.params['offline']:
            # Make the gym environment
            register_custom_envs()

            self.env = gym.make(self.params['env_name'])

            # Added by Ali
            if params['env_name'] == 'LunarLander-Customizable' and params['env_rew_weights'] is not None:
                self.env.set_rew_weights(params['env_rew_weights'])

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

            self.env.seed(seed)

            # Import plotting (locally if 'obstacles' env)
            if not (self.params['env_name'] == 'obstacles-cs285-v0'):
                import matplotlib
                matplotlib.use('Agg')

            # Maximum length for episodes
            self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps

            # Is this env continuous, or discrete?
            discrete = isinstance(self.env.action_space, gym.spaces.Discrete)

            # Are the observations images?
            img = len(self.env.observation_space.shape) > 2

            self.params['agent_params']['discrete'] = discrete

            # Observation and action sizes
            ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
            ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
            self.params['agent_params']['ac_dim'] = ac_dim
            self.params['agent_params']['ob_dim'] = ob_dim

    def run_evaluation_loop(self, n_iter, opt_policy, eval_pruner: BasePruner, buffer_path, pruning_critic=None):
        # TODO: Hard-coded
        print_period = 1

        opt_actions = []
        pruned_actions = []
        action_flags = []

        all_q_values = []
        all_rtgs = []

        if buffer_path is not None:
            # Load replay buffer data
            with open(self.params['buffer_path'], 'rb') as f:
                all_paths = pickle.load(f)
            all_paths = utils.format_reward(all_paths, self.params['env_rew_weights'])
            # Evaluate on 15% hold-out set
            _, paths = train_test_split(all_paths, test_size=0.05, random_state=self.params['seed'])

        # We run the loop only once in the MIMIC setting since we do not sample trajectories
        if self.params['offline']:
            n_iter = 1

        for itr in range(n_iter):
            if itr % print_period == 0:
                print("\n\n********** Iteration %i ************" % itr)

            # Collect trajectories
            if buffer_path is None:
                paths, envsteps_this_batch = utils.sample_trajectories(
                    self.env,
                    opt_policy,
                    min_timesteps_per_batch=self.params['batch_size'],
                    max_path_length=self.params['ep_len'],
                    render=False
                )

            # Get optimal actions and pruned_actions
            for path in tqdm(paths):
                # get actions
                opt_actions.append(path['action'].astype(int).tolist())

                # get pareto action sets per traj
                pareto_actions = eval_pruner.get_list_of_available_actions(path['observation'])
                pruned_actions.append(pareto_actions)

                # Flag if action not in pareto-set
                flags = [1 if path['action'][i] in pareto_actions[i] else 0 for i in range(len(path['action']))]
                action_flags.append(flags)

                # Get reward to go (and transform to mortality indicator, assummes Gamma == 1)
                rtg_n = utils.discounted_cumsum(path['reward'],
                                                self.params['gamma']) / 100  # to make this a mortality indicator
                rtg_n = (rtg_n + 1) / 2
                all_rtgs.append(rtg_n)

                # Get Q-values
                if pruning_critic is not None:
                    qa_values_na = pruning_critic.qa_values(path['observation'])

                    qa_values_na = ptu.from_numpy(qa_values_na)
                    ac_n = ptu.from_numpy(path['action']).to(torch.long)
                    q_values_n = torch.gather(qa_values_na, 1, ac_n.unsqueeze(1)).squeeze(1)
                    q_values_n = list(ptu.to_numpy(q_values_n))
                    # print(q_values_n)
                    all_q_values.append(q_values_n)

        # Log/save
        with open(os.path.join(self.log_dir, 'actions.pkl'), 'wb') as f:
            if pruning_critic is not None:
                pickle.dump({'opt_actions': opt_actions, 'pruned_actions': pruned_actions, 'action_flags': action_flags,
                             'mortality_rtg': all_rtgs, 'q_vals': all_q_values}, f)
            else:
                pickle.dump({'opt_actions': opt_actions, 'pruned_actions': pruned_actions, 'action_flags': action_flags,
                             'mortality_rtg': all_rtgs}, f)

        return opt_actions, pruned_actions
