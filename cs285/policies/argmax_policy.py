import numpy as np

from cs285.infrastructure.dqn_utils import get_maximizer_from_available_actions_np

class ArgMaxPolicy:
    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        # Return the action that maximizes the Q-value at the current observation as the output
        qa_values: np.ndarray = self.critic.qa_values(observation)
        ac = qa_values.argmax(axis=1)

        if len(ac.shape) > 1:
            ac = ac.squeeze()
        return ac


class PrunedArgMaxPolicy:
    def __init__(self, critic, action_pruner=None):
        self.critic = critic

        # Pruning
        self.action_pruner = action_pruner

    def get_action(self, obs: np.ndarray):
        if obs.ndim < 2:
            obs = obs[np.newaxis, :]

        qa_values: np.ndarray = self.critic.qa_values(obs)

        choose_from_pruned = False if self.action_pruner is None else True

        if choose_from_pruned:
            available_actions = self.action_pruner.get_actions(obs)

            ac = get_maximizer_from_available_actions_np(qa_values, available_actions)

        else:
            ac = qa_values.argmax(axis=1)

        if ac.ndim > 1:
            ac = ac.squeeze()
        return ac
