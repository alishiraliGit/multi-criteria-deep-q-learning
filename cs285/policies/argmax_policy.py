import numpy as np


class ArgMaxPolicy(object):

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
