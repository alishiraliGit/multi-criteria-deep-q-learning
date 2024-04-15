"""This file includes a collection of utility functions that are useful for implementing DQN."""
from collections import namedtuple
from typing import List

import numpy as np
import torch
from torch import nn
import torch.optim as optim

from rlcodebase.infrastructure.utils import pytorch_utils as ptu
from rlcodebase.infrastructure.utils.rl_utils import ConstantSchedule, PiecewiseSchedule


OptimizerSpec = namedtuple(
    'OptimizerSpec',
    ['constructor', 'optim_kwargs', 'learning_rate_schedule'],
)


###################
# Q operations
###################

def get_maximizer_from_available_actions(values_na: torch.tensor, acs_list_n: List[List[int]]) -> torch.tensor:
    """
    For each row of qa_values, returns the maximizer action from the corresponding available actions.
    @param values_na: [n x a] tensor
    @param acs_list_n: list (len n) of list of available actions
    """
    values_na = ptu.to_numpy(values_na)

    return ptu.from_numpy(get_maximizer_from_available_actions_np(values_na, acs_list_n)).to(torch.int64)


def get_maximizer_from_available_actions_np(values_na: np.ndarray, acs_list_n: List[List[int]]) -> np.ndarray:
    """
    For each row of qa_values, returns the maximizer action from the corresponding available actions.
    @param values_na: [n x a] tensor
    @param acs_list_n: list (len n) of list of available actions
    """

    return np.array([acs_list_n[idx][vals[acs_list_n[idx]].argmax()] for idx, vals in enumerate(values_na)])


def gather_by_actions(qa_values: torch.tensor, ac_n: torch.tensor) -> torch.tensor:
    if qa_values.ndim == 4:  # nare
        re_dim, ex_dim = qa_values.shape[-2:]
        ac_n = ac_n.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, 1, re_dim, ex_dim)
    elif qa_values.ndim == 3:  # nar
        re_dim = qa_values.shape[-1]
        ac_n = ac_n.unsqueeze(1).unsqueeze(2).expand(-1, 1, re_dim)
    elif qa_values.ndim == 2:  # na
        ac_n = ac_n.unsqueeze(1)

    q_values_nr = torch.gather(
        qa_values,
        1,
        ac_n
    ).squeeze(1)

    return q_values_nr


def gather_by_e(qa_values_nre: torch.tensor, ac_n: torch.tensor) -> torch.tensor:
    if qa_values_nre.ndim == 3:
        re_dim = qa_values_nre.shape[-2]
        ac_nr1 = ac_n.unsqueeze(1).unsqueeze(2).expand(-1, re_dim, 1)
    else:
        raise NotImplementedError

    q_values_nr = torch.gather(
        qa_values_nre,
        2,
        ac_nr1
    ).squeeze(2)

    return q_values_nr


###################
# Lander functions
###################

def create_lander_q_network(ob_dim, num_actions, num_rewards=1, ex_dim=1, d=64):
    if ex_dim > 1:
        output_layer = ptu.MultiDimLinear(d, (num_actions, num_rewards, ex_dim))
    else:
        if num_rewards == 1:
            output_layer = nn.Linear(d, num_actions)
        else:
            output_layer = ptu.MultiDimLinear(d, (num_actions, num_rewards))

    return nn.Sequential(
        nn.Linear(ob_dim, d),
        nn.ReLU(),
        nn.Linear(d, d),
        nn.ReLU(),
        output_layer,
    )


def lander_optimizer():
    return OptimizerSpec(
        constructor=optim.Adam,
        optim_kwargs=dict(
            lr=1,
        ),
        learning_rate_schedule=ConstantSchedule(1e-3).value,
    )


def lander_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 1),
            (num_timesteps * 0.1, 0.02),
        ], outside_value=0.02
    )


###################
# MIMIC functions
###################

def create_mimic_q_network(ob_dim, num_actions, num_rewards=1, ex_dim=1, d=64):
    if ex_dim > 1:
        output_layer = ptu.MultiDimLinear(d, (num_actions, num_rewards, ex_dim))
    else:
        if num_rewards == 1:
            output_layer = nn.Linear(d, num_actions)
        else:
            output_layer = ptu.MultiDimLinear(d, (num_actions, num_rewards))

    return nn.Sequential(
        nn.Linear(ob_dim, d),
        nn.ReLU(),
        nn.Linear(d, d),
        nn.ReLU(),
        output_layer,
    )


def mimic_optimizer():
    return OptimizerSpec(
        constructor=optim.Adam,
        optim_kwargs=dict(
            lr=1,
        ),
        # TODO: Hard-coded
        learning_rate_schedule=ConstantSchedule(1e-4).value,  # 1e-4
    )


###################
# SepsisSim functions
###################
def sepsissim_optimizer():
    return OptimizerSpec(
        constructor=optim.Adam,
        optim_kwargs=dict(
            lr=1,
        ),
        learning_rate_schedule=ConstantSchedule(1e-4).value,
    )
