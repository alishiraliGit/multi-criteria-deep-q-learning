import numpy as np

from .base_critic import BaseCritic

from cs285.critics.dqn_critic import DQNCritic
from cs285.critics.cql_critic import CQLCritic
from cs285.infrastructure import pytorch_util as ptu


class ParetoOptDQNCritic(BaseCritic):
    def __init__(self, dqn_critics=None, saved_dqn_critics_paths=None, **kwargs):
        super().__init__(**kwargs)

        if dqn_critics is not None:
            self.dqn_critics = dqn_critics
        elif saved_dqn_critics_paths is not None:
            self.dqn_critics = [DQNCritic.load(f) for f in saved_dqn_critics_paths]
        else:
            raise Exception('Neither DQN critics nor paths to DQN critic is provided!')

        # Check all critics have similar dimensions
        assert len(set([dqn_critic.ob_dim for dqn_critic in self.dqn_critics])) == 1
        assert len(set([dqn_critic.ac_dim for dqn_critic in self.dqn_critics])) == 1

        #Set all critics to device
        for dqn_critic in self.dqn_critics:
            dqn_critic.q_net.to(ptu.device)
            dqn_critic.q_net_target.to(ptu.device)

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        pass

    def qa_values(self, ob_no: np.ndarray):
        if ob_no.ndim < 2:
            ob_no = ob_no[np.newaxis, :]

        qa_values_nac: np.ndarray = np.stack([dqn_critic.qa_values(ob_no) for dqn_critic in self.dqn_critics], axis=-1)

        return qa_values_nac

class ParetoOptCQLCritic(BaseCritic):
    def __init__(self, dqn_critics=None, saved_dqn_critics_paths=None, **kwargs):
        super().__init__(**kwargs)

        if dqn_critics is not None:
            self.dqn_critics = dqn_critics
        elif saved_dqn_critics_paths is not None:
            self.dqn_critics = [CQLCritic.load(f) for f in saved_dqn_critics_paths]
        else:
            raise Exception('Neither CQL critics nor paths to CQL critic is provided!')

        # Check all critics have similar dimensions
        assert len(set([dqn_critic.ob_dim for dqn_critic in self.dqn_critics])) == 1
        assert len(set([dqn_critic.ac_dim for dqn_critic in self.dqn_critics])) == 1

        #Set all critics to device
        for dqn_critic in self.dqn_critics:
            dqn_critic.q_net.to(ptu.device)
            dqn_critic.q_net_target.to(ptu.device)

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        pass

    def qa_values(self, ob_no: np.ndarray):
        if ob_no.ndim < 2:
            ob_no = ob_no[np.newaxis, :]

        qa_values_nac: np.ndarray = np.stack([dqn_critic.qa_values(ob_no) for dqn_critic in self.dqn_critics], axis=-1)

        return qa_values_nac
