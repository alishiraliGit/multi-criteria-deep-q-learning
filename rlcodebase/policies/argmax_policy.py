import numpy as np

from rlcodebase.policies.base_policy import BasePolicy
from rlcodebase.pruners.base_pruner import BasePruner
from rlcodebase.infrastructure.utils.dqn_utils import get_maximizer_from_available_actions_np


class ArgMaxPolicy(BasePolicy):
    def __init__(self, critic):
        self.critic = critic

    def get_actions(self, ob_no: np.ndarray) -> np.ndarray:
        if ob_no.ndim < 2:
            ob_no = ob_no[np.newaxis, :]

        qa_values_na: np.ndarray = self.critic.qa_values(ob_no)

        ac_n = qa_values_na.argmax(axis=1)

        return ac_n

    def update(self, *args, **kwargs):
        pass


class PrunedArgMaxPolicy(BasePolicy):
    def __init__(self, critic, action_pruner: BasePruner):
        self.critic = critic

        # Pruning
        self.action_pruner = action_pruner

    def get_actions(self, ob_no: np.ndarray) -> np.ndarray:
        if ob_no.ndim < 2:
            ob_no = ob_no[np.newaxis, :]

        qa_values_na: np.ndarray = self.critic.qa_values(ob_no)

        available_acs_n = self.action_pruner.get_list_of_available_actions(ob_no)

        ac_n = get_maximizer_from_available_actions_np(qa_values_na, available_acs_n)

        return ac_n

    def update(self, *args, **kwargs):
        pass
