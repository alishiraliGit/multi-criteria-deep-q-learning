import abc
import numpy as np


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def get_actions(self, ob_no: np.ndarray) -> np.ndarray:
        pass

    # For compatibility
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return self.get_actions(obs)[0]

    @abc.abstractmethod
    def update(self, obs: np.ndarray, acs: np.ndarray, **kwargs) -> dict:
        pass
