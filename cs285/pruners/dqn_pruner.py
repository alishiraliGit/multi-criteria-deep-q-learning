from typing import List

import numpy as np

from cs285.pruners.base_pruner import BasePruner
from cs285.pruners.primary_pruner import RandomPruner
from cs285.critics.dqn_critic import MDQNCritic, ExtendedMDQNCritic


class MDQNPruner(BasePruner):
    def __init__(self, n_draw, file_path=None, critic=None):
        assert (file_path is not None) ^ (critic is not None)
        if file_path is not None:
            self.mdqn_critic = MDQNCritic.load(file_path)
        else:
            self.mdqn_critic = critic

        actor = self.mdqn_critic.get_actor_class()(self.mdqn_critic)
        self.random_pruner = RandomPruner(actor, n_draw=n_draw)

    def get_list_of_available_actions(self, ob_no: np.ndarray) -> List[List[int]]:
        return self.random_pruner.get_list_of_available_actions(ob_no)


class ExtendedMDQNPruner(BasePruner):
    def __init__(self, n_draw, file_path=None, critic=None):
        assert (file_path is not None) ^ (critic is not None)
        if file_path is not None:
            self.emdqn_critic = ExtendedMDQNCritic.load(file_path)
        else:
            self.emdqn_critic = critic

        actor = self.emdqn_critic.get_actor_class()(self.emdqn_critic)
        self.random_pruner = RandomPruner(actor, n_draw=n_draw)

    def get_list_of_available_actions(self, ob_no: np.ndarray) -> List[List[int]]:
        return self.random_pruner.get_list_of_available_actions(ob_no)
