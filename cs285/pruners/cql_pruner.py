from typing import List

import numpy as np

from cs285.critics.independent_dqns_critic import ICQLCritic
from cs285.pruners.primary_pruner import ParetoOptimalPruner
from cs285.pruners.base_pruner import BasePruner


class ICQLPruner(BasePruner):
    def __init__(self, file_paths, pruning_eps, **kwargs):
        super().__init__(**kwargs)

        self.icql_critic = ICQLCritic(saved_dqn_critics_paths=file_paths)
        self.pareto_opt_pruner = ParetoOptimalPruner(self.icql_critic, eps=pruning_eps)

    def get_list_of_available_actions(self, ob_no: np.ndarray) -> List[List[int]]:
        return self.pareto_opt_pruner.get_list_of_available_actions(ob_no)
