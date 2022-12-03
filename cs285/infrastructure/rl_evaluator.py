import pickle
import os
from cs285.infrastructure.atari_wrappers import ReturnWrapper

import gym
from gym import wrappers
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure import utils
from cs285.infrastructure.dqn_utils import register_custom_envs


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
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        # To be set later in run_training_loop
        self.total_envsteps = None
        self.start_time = None

        #############
        # ENV
        #############

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
        if not(self.params['env_name'] == 'obstacles-cs285-v0'):
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

    def run_evaluation_loop(self, n_iter, collect_policy, eval_policy):
        # TODO: Hard-coded
        print_period = 1

        opt_actions = []
        pareto_opt_actions = []

        for itr in range(n_iter):
            if itr % print_period == 0:
                print("\n\n********** Iteration %i ************" % itr)

            # Collect trajectories
            paths, envsteps_this_batch = utils.sample_trajectories(
                self.env,
                collect_policy,
                min_timesteps_per_batch=self.params['batch_size'],
                max_path_length=self.params['ep_len'],
                render=False
            )

            #
            for path in paths:
                opt_actions.append(path['action'].astype(int).tolist())

                pareto_opt_actions.append([eval_policy.get_action(ob) for ob in path['observation']])

        # Log/save
        with open(os.path.join(self.log_dir, 'actions.pkl'), 'wb') as f:
            pickle.dump({'opt_actions': opt_actions, 'pareto_opt_actions': pareto_opt_actions}, f)

        return opt_actions, pareto_opt_actions
