import warnings

import numpy as np


class ParetoOptimalPolicy(object):

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
                return ParetoOptimalPolicy.find_pareto_optimal_actions(action_values)
            else:
                return ParetoOptimalPolicy.find_strong_pareto_optimal_actions(action_values, eps/2, n_rept - 1)
        else:
            return pareto_actions

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]

        qa_values_ac: np.ndarray = self.critic.qa_values(observation)[0]

        return self.find_strong_pareto_optimal_actions(qa_values_ac, self.eps)


if __name__ == '__main__':
    # Test find_pareto_optimal_actions
    values_ = np.array([
        [1, 1.2, 1, 1],
        [2, 1, 2, 2],
        [2, 1.1, 2, 2]
    ])

    print(ParetoOptimalPolicy.find_strong_pareto_optimal_actions(values_, eps=0.1))
