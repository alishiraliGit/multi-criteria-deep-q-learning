import numpy as np
import gym
from gym import spaces

from ext.gumbelmaxscmsimv2.sepsisSimDiabetes.MDP import MDP
from ext.gumbelmaxscmsimv2.sepsisSimDiabetes.MDP import Action


class SepsisSimWrapper(gym.Env):
    def __init__(self, p_diabetes=0.2, **_kwargs):
        self.p_diabetes = p_diabetes

        self.observation_space = spaces.Box(0, 4, shape=(7,))

        self.action_space = spaces.Discrete(Action.NUM_ACTIONS_TOTAL)

        self.reward_dim = 5
        self.episode_final_rewards = []

        self.curr_step = None

        self._reset()

    def _seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def _reset(self):
        self.curr_step = 0
        self.mdp = MDP(init_state_idx=None, policy_array=None, p_diabetes=self.p_diabetes)

        return self.mdp.state.get_state_vector().astype(float)

    def _step(self, action):
        # Transition
        rew_final = self.mdp.transition(Action(action_idx=action))
        s1 = self.mdp.state.get_state_vector().astype(float)

        # Get the rewards
        rews = np.zeros((5,))
        rews[4] = rew_final * 100

        hr_state, sysbp_state, percoxyg_state, glucose_state, antibiotic_state, vaso_state, vent_state = s1

        if hr_state != 1:
            rews[0] = -1
        if sysbp_state != 1:
            rews[1] = -1
        if percoxyg_state != 1:
            rews[2] = -1
        if glucose_state != 2:
            rews[3] = -1

        # Is it done?
        done = self.mdp.state.check_absorbing_state()

        if done:
            self.episode_final_rewards.append(rews[-1])

        self.curr_step += 1

        info = {}  # Dummy

        return np.array(s1), rews, done, info

    def reset(self, **_kwargs):
        return self._reset()

    def step(self, *args, **kwargs):
        return self._step(*args, **kwargs)

    def careless_step(self, *args, **kwargs):
        return self.step(*args, **kwargs)
