import pickle
import os
from typing import List, Union

import numpy as np
import torch

from rlcodebase.infrastructure.utils import rl_utils
from rlcodebase.infrastructure.utils import pytorch_utils as ptu
from rlcodebase.infrastructure.utils.dqn_utils import gather_by_actions
from rlcodebase.infrastructure.utils.general_utils import merge_lists
from rlcodebase.envs.gym_utils import init_gym_and_update_params
from rlcodebase.envs.mimic_utils import init_mimic_and_update_params
from rlcodebase.eval import metrics
from rlcodebase.pruners.base_pruner import BasePruner
from rlcodebase.policies.base_policy import BasePolicy
from rlcodebase.critics.dqn_critic import DQNCritic

from tqdm import tqdm


class RLEvaluator(object):

    def __init__(self, params):

        #############
        # INIT
        #############
        # Get params
        self.params = params

        # Logging
        self.log_dir = self.params['logdir']
        self.log_freq = self.params['log_freq']

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)

        #############
        # ENV
        #############
        self.offline = params['offline']
        self.params['agent_params']['offline'] = params['offline']

        if not self.offline:
            self.env = init_gym_and_update_params(params)
        else:
            self.env, self.train_paths, self.test_paths = init_mimic_and_update_params(params)

    def _eval(
            self,
            path,
            eval_policy: Union[BasePolicy, None],
            eval_critic: Union[DQNCritic, None],
            eval_pruner: Union[BasePruner, None]):

        opt_observations, opt_actions, opt_terminals, opt_rewards = \
            path['observation'], path['action'], path['terminal'], path['reward']

        n = len(opt_observations)

        # Get reward-to-go
        opt_rtgs = rl_utils.discounted_cumsum(opt_rewards, self.params['gamma'])

        # Get available actions and check whether the optimal actions are available
        if eval_pruner is not None:
            available_actions = eval_pruner.get_list_of_available_actions(opt_observations)

            # Flag if the optimal action is in pruned action set
            opt_acts_are_available = [
                1 if opt_a in available_actions[t] else 0
                for t, opt_a in enumerate(opt_actions)
            ]
        else:
            available_actions = [None]*n
            opt_acts_are_available = [None]*n

        # Get eval. policy actions
        if eval_policy is not None:
            eval_actions = eval_policy.get_actions(opt_observations)
        else:
            eval_actions = [None]*n

        # Get the eval. Q-values of the optimal actions
        if eval_critic is not None:
            eval_qa_values_na = eval_critic.qa_values(opt_observations)

            eval_q_values = ptu.to_numpy(gather_by_actions(
                ptu.from_numpy(eval_qa_values_na),
                ptu.from_numpy(opt_actions).to(torch.long)
            ))
        else:
            eval_q_values = [None]*n

        return opt_observations, opt_actions, opt_terminals, opt_rewards, opt_rtgs, \
            available_actions, opt_acts_are_available, eval_actions, eval_q_values

    def run_evaluation_loop(
            self,
            n_iter,
            opt_policy: Union[BasePolicy, None],
            eval_policy: Union[BasePolicy, None],
            eval_critic: Union[DQNCritic, None],
            eval_pruner: Union[BasePruner, None],
            eval_metrics: List[metrics.EvalMetricBase],
            ignore_metrics: bool = False):

        # We run the loop only once in the offline setting since we do not sample trajectories
        if self.offline:
            n_iter = 1

        list_opt_observations = []
        list_opt_actions = []
        list_opt_terminals = []
        list_opt_rewards = []
        list_opt_rtgs = []

        list_available_actions = []
        list_opt_acts_are_available = []
        list_eval_actions = []
        list_eval_q_values = []

        for itr in range(n_iter):
            if itr % self.log_freq == 0:
                print(f'\n\n********** Iteration {itr} ************')

            # Collect trajectories
            if not self.offline:
                test_paths, envsteps_this_batch = rl_utils.sample_trajectories(
                    self.env, opt_policy,
                    min_timesteps_per_batch=self.params['eval_batch_size'],
                    max_path_length=self.params['ep_len'],
                    render=False
                )
            else:
                test_paths = self.test_paths

            # Preprocess trajectories
            for path in tqdm(test_paths):
                opt_observations, opt_actions, opt_terminals, opt_rewards, opt_rtgs, \
                    available_actions, opt_acts_are_available, eval_actions, eval_q_values \
                    = self._eval(path, eval_policy, eval_critic, eval_pruner)

                list_opt_observations.append(opt_observations)
                list_opt_actions.append(opt_actions)
                list_opt_terminals.append(opt_terminals)
                list_opt_rewards.append(opt_rewards)
                list_opt_rtgs.append(opt_rtgs)

                list_available_actions.append(available_actions)
                list_opt_acts_are_available.append(opt_acts_are_available)
                list_eval_actions.append(eval_actions)
                list_eval_q_values.append(eval_q_values)

        actions_dict = {
            'opt_actions': list_opt_actions,
            'policy_actions': list_eval_actions,
            'pruned_actions': list_available_actions,
            'action_flags': list_opt_acts_are_available,
            'mortality_rtg': list_opt_rtgs,
            'q_vals': list_eval_q_values,
        }

        with open(os.path.join(self.log_dir, 'actions.pkl'), 'wb') as f:
            pickle.dump(actions_dict, f)

        if ignore_metrics:
            return None, actions_dict

        # Calc. metrics
        flat_opt_observations = np.concatenate(list_opt_observations, axis=0)
        flat_eval_actions = np.concatenate(list_eval_actions, axis=0)
        flat_opt_terminals = np.concatenate(list_opt_terminals, axis=0)
        flat_opt_rewards = np.concatenate(list_opt_rewards, axis=0)
        flat_opt_rtgs = np.concatenate(list_opt_rtgs, axis=0)

        flat_available_actions = merge_lists(list_available_actions)
        flat_eval_q_values = np.concatenate(list_eval_q_values, axis=0)
        flat_opt_actions = np.concatenate(list_opt_actions, axis=0)

        # Calc. metrics
        metric_values = {}
        for metric in eval_metrics:
            if metric.requires_fitting:
                if not self.offline:
                    raise NotImplementedError
                metric.fit(self.train_paths, self.test_paths, self.params, eval_policy)

            metric_values[metric.name] = metric.value(
                flat_opt_observations, flat_available_actions, flat_eval_actions, flat_opt_rewards, flat_opt_terminals,
                flat_opt_rtgs, flat_eval_q_values, flat_opt_actions)

        print(metric_values)

        with open(os.path.join(self.log_dir, 'metrics.pkl'), 'wb') as f:
            pickle.dump(metric_values, f)

        return metric_values, actions_dict
