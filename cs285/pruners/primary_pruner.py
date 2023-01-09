import warnings
from typing import List

import numpy as np

from cs285.policies.base_policy import BasePolicy
from cs285.pruners.base_pruner import BasePruner


class RandomPruner(BasePruner):
    def __init__(self, actor: BasePolicy, n_draw=10):
        self.actor = actor
        self.n_draw = n_draw

    def get_list_of_available_actions(self, ob_no: np.ndarray) -> List[List[int]]:
        if ob_no.ndim < 2:
            ob_no = ob_no[np.newaxis, :]

        n = ob_no.shape[0]

        ac_nd = np.zeros((n, self.n_draw)).astype(int)
        for cnt in range(self.n_draw):
            ac_nd[:, cnt] = self.actor.get_actions(ob_no)

        acs_n = [list(set(ac_d.tolist())) for ac_d in ac_nd]

        return acs_n


class ParetoOptimalPruner(BasePruner):

    def __init__(self, critic, eps=0):
        self.critic = critic

        self.eps = eps

    @staticmethod
    def find_pareto_optimal_actions(action_values: np.ndarray):
        n_action = action_values.shape[0]

        pareto_actions = []
        for a in range(n_action):
            ge_a_mask = np.all(action_values[a:(a + 1)] <= action_values, axis=1)

            if np.any(action_values[a:(a+1)] < action_values[ge_a_mask]):
                continue

            pareto_actions.append(a)

        return pareto_actions

    @staticmethod
    def find_strong_pareto_optimal_actions(action_values: np.ndarray, eps, n_rept=5):
        n_action = action_values.shape[0]

        pareto_actions = []
        for a in range(n_action):
            if np.sum(np.mean(action_values[a:(a + 1)] > action_values, axis=1) > eps) == n_action - 1:
                pareto_actions.append(a)

        if len(pareto_actions) == 0:
            if n_rept == 0:
                warnings.warn('No strong Pareto optimal action found. Returning Pareto optimal actions.')
                return ParetoOptimalPruner.find_pareto_optimal_actions(action_values)
            else:
                return ParetoOptimalPruner.find_strong_pareto_optimal_actions(action_values, eps / 2, n_rept - 1)
        else:
            return pareto_actions

    def get_list_of_available_actions(self, ob_no: np.ndarray) -> List[List[int]]:
        if ob_no.ndim < 2:
            ob_no = ob_no[np.newaxis, :]

        qa_values_nac: np.ndarray = self.critic.qa_values(ob_no)

        return [self.find_strong_pareto_optimal_actions(vals, self.eps) for vals in qa_values_nac]


if __name__ == '__main__':
    # Test find_pareto_optimal_actions
    values_ = np.array([
        [1, 1, 1, 1],
        [0, 0, 0, 2],
        [2, 2, 2, 0]
    ])

    eps_ = 0.3

    print(ParetoOptimalPruner.find_strong_pareto_optimal_actions(values_, eps=eps_))
