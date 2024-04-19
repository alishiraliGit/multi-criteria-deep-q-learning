import numpy as np

from rlcodebase.policies.base_policy import BasePolicy
from rlcodebase.pruners.base_pruner import BasePruner
from rlcodebase.infrastructure.utils.dqn_utils import get_maximizer_from_available_actions_np
from rlcodebase.infrastructure.utils import pytorch_utils as ptu

import torch.optim as optim
from torch.nn import utils
from torch import nn
import torch.nn.functional as F
import torch

class BCQPolicy(BasePolicy):
    def __init__(self, critic, threshold=0.3, eval_eps=1e-3, **kwargs):
        self.critic = critic
        self.threshold = threshold
        self.eval_eps = eval_eps
    
    def get_actions(self, ob_no: np.ndarray, eval=False):
		# Select action according to policy with probability (1-eps)
		# otherwise, select random action
        if ob_no.ndim < 2:
            ob_no = ob_no[np.newaxis, :]
        if np.random.uniform(0,1) > self.eval_eps:
            with torch.no_grad():
                # print(ob_no.shape)
                # state = torch.FloatTensor(ob_no).reshape(self.critic.ob_dim).to(ptu.device)
                q, imt, i = self.critic.qa_values(ob_no)
                imt = imt.exp()
                imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()
                q, imt, i = ptu.to_numpy(q), ptu.to_numpy(imt), ptu.to_numpy(i)
				# Use large negative number to mask actions from argmax
                print((imt * q + (1. - imt) * -1e8).argmax(1))
                return np.array((imt * q + (1. - imt) * -1e8).argmax(1))
        else:
            return np.random.randint(self.critic.ac_dim)

    # def get_actions(self, ob_no: np.ndarray) -> np.ndarray:
    #     if ob_no.ndim < 2:
    #         ob_no = ob_no[np.newaxis, :]
    #     qa_values_na: np.ndarray = self.critic.qa_values(ob_no)
    #     ac_n = qa_values_na.argmax(axis=1)
    #     return ac_n

    def update(self, *args, **kwargs):
        pass