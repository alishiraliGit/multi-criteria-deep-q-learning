import abc
from typing import List
import numpy as np


class BasePruner(abc.ABC):
    @abc.abstractmethod
    def get_list_of_available_actions(self, ob_no: np.ndarray) -> List[List[int]]:
        pass
