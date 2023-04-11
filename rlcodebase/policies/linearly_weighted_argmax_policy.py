import numpy as np
import torch
from torch.nn.functional import softmax

from rlcodebase.policies.base_policy import BasePolicy
from rlcodebase.infrastructure.utils import pytorch_utils as ptu


def draw_w(size, b) -> np.ndarray:
    if isinstance(b, int) or isinstance(b, float) or len(b) == 1:
        w = np.random.random(size) * b + 1
    else:
        raise NotImplementedError
        # return 0.5*np.array(b)[np.newaxis, :] + np.random.random(size) * np.array(b)[np.newaxis, :]

    # Normalize across last dimension
    w = w/np.sum(w, axis=-1, keepdims=True)

    return w


class LinearlyWeightedArgMaxPolicy(BasePolicy):
    def __init__(self, critic):
        self.critic = critic

    @staticmethod
    def get_actions_for_w(qa_values_nar, w_nr):
        # Works for both numpy arrays and torch tensors
        if isinstance(w_nr, np.ndarray):
            w_n1r = np.expand_dims(w_nr, 1)
        else:
            w_n1r = w_nr.unsqueeze(1)

        qa_values_na = (qa_values_nar * w_n1r).sum(2)

        return qa_values_na.argmax(1)

    def get_actions(self, ob_no: np.ndarray):
        if ob_no.ndim < 2:
            ob_no = ob_no[np.newaxis, :]

        n = ob_no.shape[0]

        qa_values_nar: np.ndarray = self.critic.qa_values(ob_no)

        w_nr = draw_w((n, self.critic.re_dim), self.critic.b)

        return self.get_actions_for_w(qa_values_nar, w_nr)

    def update(self, *args, **kwargs):
        pass


class LinearlyWeightedSoftmaxPolicy(BasePolicy):
    def __init__(self, critic):
        self.critic = critic

    @staticmethod
    def get_actions_for_w(qa_values_nar, w_nr, alpha):
        is_numpy = isinstance(qa_values_nar, np.ndarray)
        if is_numpy:
            qa_values_nar = ptu.from_numpy(qa_values_nar)
            w_nr = ptu.from_numpy(w_nr)

        qa_values_na = (qa_values_nar * w_nr.unsqueeze(1)).sum(dim=2)

        p_na = softmax(qa_values_na * alpha, dim=1)

        dist = torch.distributions.categorical.Categorical(p_na)

        ac_n = dist.sample()

        if is_numpy:
            ac_n = ptu.to_numpy(ac_n)
        return ac_n

    def get_actions(self, ob_no: np.ndarray):
        if ob_no.ndim < 2:
            ob_no = ob_no[np.newaxis, :]

        n = ob_no.shape[0]

        qa_values_nar: np.ndarray = self.critic.qa_values(ob_no)

        w_nr = draw_w((n, self.critic.re_dim), self.critic.b)

        return self.get_actions_for_w(qa_values_nar, w_nr, self.critic.alpha)

    def update(self, *args, **kwargs):
        pass


class ExtendedLinearlyWeightedArgMaxPolicy(BasePolicy):
    def __init__(self, critic):
        self.critic = critic

    @staticmethod
    def get_actions_for_w(qa_values_nare, w_nr):
        # Works for both numpy arrays and torch tensors
        if isinstance(qa_values_nare, np.ndarray):
            w_n1r1 = np.expand_dims(w_nr, (1, 3))
            qa_values_na = (qa_values_nare * w_n1r1).sum(axis=2).max(axis=2)
        elif isinstance(qa_values_nare, torch.Tensor):
            w_n1r1 = w_nr.unsqueeze(1).unsqueeze(3)
            qa_values_na, _ = (qa_values_nare * w_n1r1).sum(dim=2).max(dim=2)
        else:
            raise Exception('Invalid input!')

        return qa_values_na.argmax(1)

    def get_actions(self, ob_no: np.ndarray):
        if ob_no.ndim < 2:
            ob_no = ob_no[np.newaxis, :]

        n = ob_no.shape[0]

        qa_values_nare: np.ndarray = self.critic.qa_values(ob_no)

        w_nr = draw_w((n, self.critic.re_dim), self.critic.b)

        return self.get_actions_for_w(qa_values_nare, w_nr)

    def update(self, *args, **kwargs):
        pass


class ExtendedLinearlyWeightedSoftmaxPolicy(BasePolicy):

    def __init__(self, critic):
        self.critic = critic

    @staticmethod
    def get_actions_for_w(qa_values_nare, w_nr, alpha):
        is_numpy = isinstance(qa_values_nare, np.ndarray)
        if is_numpy:
            qa_values_nare = ptu.from_numpy(qa_values_nare)
            w_nr = ptu.from_numpy(w_nr)

        w_n1r1 = w_nr.unsqueeze(1).unsqueeze(3)
        qa_values_na, _ = (qa_values_nare * w_n1r1).sum(dim=2).max(dim=2)

        p_na = softmax(qa_values_na * alpha, dim=1)

        dist = torch.distributions.categorical.Categorical(p_na)

        ac_n = dist.sample()

        if is_numpy:
            ac_n = ptu.to_numpy(ac_n)
        return ac_n

    def get_actions(self, ob_no: np.ndarray):
        if ob_no.ndim < 2:
            ob_no = ob_no[np.newaxis, :]

        n = ob_no.shape[0]

        qa_values_nare: np.ndarray = self.critic.qa_values(ob_no)

        w_nr = draw_w((n, self.critic.re_dim), self.critic.b)

        # noinspection PyTypeChecker
        return self.get_actions_for_w(qa_values_nare, w_nr, self.critic.alpha)

    def update(self, *args, **kwargs):
        pass
