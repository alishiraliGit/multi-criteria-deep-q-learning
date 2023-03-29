import pickle
import os
from typing import List, Union

import numpy as np
import torch

from rlcodebase.infrastructure.utils import rl_utils
from rlcodebase.infrastructure.utils import pytorch_utils as ptu
from rlcodebase.infrastructure.utils.dqn_utils import gather_by_actions
from rlcodebase.envs.gym_utils import init_gym_and_update_params
from rlcodebase.envs.mimic_utils import init_mimic_and_update_params
from rlcodebase.eval import metrics
from rlcodebase.pruners.base_pruner import BasePruner
from rlcodebase.policies.base_policy import BasePolicy
from rlcodebase.critics.dqn_critic import DQNCritic


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

        # To be set later in run_training_loop
        self.total_envsteps = None
        self.start_time = None

        #############
        # ENV
        #############
        self.offline = params['offline']
        self.params['agent_params']['offline'] = params['offline']

        if not self.offline:
            self.env = init_gym_and_update_params(params)
        else:
            self.env, self.train_paths, self.test_paths = init_mimic_and_update_params(params)

    def run_evaluation_loop(self,
                            n_iter,
                            opt_policy: Union[BasePolicy, None],
                            eval_policy: Union[BasePolicy, None],
                            eval_critic: Union[DQNCritic, None],
                            eval_pruner: Union[BasePruner, None],
                            eval_metrics: List[metrics.EvalMetricBase]):
        # We run the loop only once in the MIMIC setting since we do not sample trajectories
        if self.offline:
            n_iter = 1

        all_observations = []
        all_available_actions = []
        all_actions = []
        all_rewards = []
        all_terminals = []
        all_rtgs = []
        all_q_values = []
        all_opt_actions = []
        for itr in range(n_iter):
            if itr % self.log_freq == 0:
                print("\n\n********** Iteration %i ************" % itr)

            # Collect trajectories
            if not self.offline:
                test_paths, envsteps_this_batch = rl_utils.sample_trajectories(
                    self.env, opt_policy,
                    min_timesteps_per_batch=self.params['batch_size'],
                    max_path_length=self.params['ep_len'],
                    render=False
                )
            else:
                test_paths = self.test_paths

            observations, opt_actions, _, terminals, rewards, _ = rl_utils.convert_listofrollouts(test_paths)

            # Get available actions
            if eval_pruner is not None:
                available_actions = eval_pruner.get_list_of_available_actions(observations)
            else:
                available_actions = [None]

            # Get policy actions
            actions = eval_policy.get_actions(observations)

            # Get the Q-values
            if eval_critic is not None:
                qa_values_na = eval_critic.qa_values(observations)

                q_values = ptu.to_numpy(gather_by_actions(
                    ptu.from_numpy(qa_values_na),
                    ptu.from_numpy(opt_actions).to(torch.long)
                ))
            else:
                q_values = None

            # Get reward-to-go
            rtgs = []
            for path in test_paths:
                re_n = path['reward']

                rtg_n = rl_utils.discounted_cumsum(re_n, self.params['gamma'])

                rtgs.append(rtg_n)
            rtgs = np.concatenate(rtgs, axis=0)

            # Append
            all_observations.append(observations)
            all_available_actions.extend(available_actions)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_terminals.append(terminals)
            all_rtgs.append(rtgs)
            all_q_values.append(q_values)
            all_opt_actions.append(opt_actions.astype(int))

        # Concat
        all_observations = np.concatenate(all_observations, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        all_rewards = np.concatenate(all_rewards, axis=0)
        all_terminals = np.concatenate(all_terminals, axis=0)
        all_rtgs = np.concatenate(all_rtgs, axis=0)
        all_q_values = np.concatenate(all_q_values, axis=0)
        all_opt_actions = np.concatenate(all_opt_actions, axis=0)

        # Calc. metrics
        metric_values = {}
        for metric in eval_metrics:
            if metric.requires_fitting:
                if not self.offline:
                    raise NotImplementedError
                metric.fit(self.train_paths, self.test_paths, self.params, eval_policy)

            metric_values[metric.name] = metric.value(all_observations, all_available_actions, all_actions, all_rewards,
                                                      all_terminals, all_rtgs, all_q_values, all_opt_actions)

        actions_dict = {
            'opt_actions': all_opt_actions,
            'policy_actions': all_actions,
        }

        print(metric_values)

        with open(os.path.join(self.log_dir, 'actions.pkl'), 'wb') as f:
            pickle.dump(actions_dict, f)

        with open(os.path.join(self.log_dir, 'metrics.pkl'), 'wb') as f:
            pickle.dump(metric_values, f)

        return metric_values, actions_dict
