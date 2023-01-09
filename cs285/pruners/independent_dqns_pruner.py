from typing import List

import numpy as np

from cs285.pruners.base_pruner import BasePruner
from cs285.pruners.primary_pruner import ParetoOptimalPruner
from cs285.critics.independent_dqns_critic import IDQNCritic


class IDQNPruner(BasePruner):
    def __init__(self, pruning_eps, dqn_critics=None, saved_dqn_critics_paths=None):
        self.idqn_critic = IDQNCritic(dqn_critics, saved_dqn_critics_paths)
        self.pareto_opt_pruner = ParetoOptimalPruner(self.idqn_critic, eps=pruning_eps)

    def get_list_of_available_actions(self, ob_no: np.ndarray) -> List[List[int]]:
        return self.pareto_opt_pruner.get_list_of_available_actions(ob_no)
