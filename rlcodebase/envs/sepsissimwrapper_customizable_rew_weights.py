import numpy as np

from rlcodebase.envs.sepsissimwrapper_multi_rew import SepsisSimWrapper as SepsisSimWrapperMultiRew


class SepsisSimWrapper(SepsisSimWrapperMultiRew):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.multi_rew_dim = self.reward_dim
        self.reward_dim = 1

        # Reward weights
        self.inter_rew_weights = np.zeros((self.multi_rew_dim - 1,))
        self.final_rew_weight = 1

    def set_rew_weights(self, rew_weights):
        assert len(rew_weights) == self.multi_rew_dim

        self.inter_rew_weights = rew_weights[:self.multi_rew_dim - 1]
        self.final_rew_weight = rew_weights[-1]

    def step(self, *args, **kwargs):
        s1, rews, done, info = self._step(*args, **kwargs)

        inter_rews, final_rew = rews[:-1], rews[-1]

        rew = final_rew * self.final_rew_weight
        for i_rew, inter_rew in enumerate(inter_rews):
            rew += inter_rew * self.inter_rew_weights[i_rew]

        return s1, rew, done, info

