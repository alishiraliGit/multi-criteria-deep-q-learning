import warnings

import numpy as np
from scipy.spatial.distance import pdist, squareform


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

    @staticmethod
    def find_strong_pareto_optimal_actions_scipy(action_values: np.ndarray, eps, n_rept=5):
        """
        Note: This shorter implementation is slower than the original implementation.
        @param action_values:
        @param eps:
        @param n_rept:
        @return:
        """
        n_action = action_values.shape[0]

        u_g_v = squareform(pdist(action_values, lambda u, v: np.mean(u > v)))  # u > v
        u_g_v = np.triu(u_g_v, k=1) + np.tril(1 - u_g_v, k=-1)

        pareto_actions = np.where(np.sum(u_g_v > eps, axis=1) == n_action - 1)[0].tolist()

        if len(pareto_actions) == 0:
            if n_rept == 0:
                warnings.warn('No strong Pareto optimal action found. Returning Pareto optimal actions.')
                return ParetoOptimalPolicy.find_pareto_optimal_actions(action_values)
            else:
                return ParetoOptimalPolicy.find_strong_pareto_optimal_actions_scipy(action_values, eps / 2, n_rept - 1)
        else:
            return pareto_actions

    def get_action(self, obs: np.ndarray):
        if obs.ndim < 2:
            obs = obs[np.newaxis, :]

        qa_values_ac: np.ndarray = self.critic.qa_values(obs)[0]

        return self.find_strong_pareto_optimal_actions(qa_values_ac, self.eps)

    def get_actions(self, ob_no: np.ndarray):
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

    n_rept_ = 10000

    import time

    tic = time.time()
    for _ in range(n_rept_):
        values_ = np.random.random((6, 100))
        a_ = ParetoOptimalPolicy.find_strong_pareto_optimal_actions(values_, eps=eps_)
    toc = time.time()
    print(toc - tic)

    tic = time.time()
    for _ in range(n_rept_):
        values_ = np.random.random((6, 100))
        b_ = ParetoOptimalPolicy.find_strong_pareto_optimal_actions_scipy(values_, eps=eps_)
    toc = time.time()
    print(toc - tic)

    print(ParetoOptimalPolicy.find_strong_pareto_optimal_actions(values_, eps=eps_))
    print(ParetoOptimalPolicy.find_strong_pareto_optimal_actions_scipy(values_, eps=eps_))
