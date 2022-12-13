import warnings

from cs285.agents.base_agent import BaseAgent
from cs285.policies.pareto_opt_policy import ParetoOptimalPolicy
from cs285.critics.pareto_opt_dqn_critic import ParetoOptDQNCritic
from cs285.critics.dqn_critic import MDQNCritic


class LoadedParetoOptDQNAgent(BaseAgent):
    def __init__(self, file_paths, pruning_eps, **kwargs):
        super().__init__(**kwargs)

        self.critic = ParetoOptDQNCritic(saved_dqn_critics_paths=file_paths)
        self.actor = ParetoOptimalPolicy(self.critic, eps=pruning_eps)

    def train(self) -> dict:
        pass

    def add_to_replay_buffer(self, paths):
        pass

    def sample(self, batch_size):
        pass

    def save(self, path):
        pass


class LoadedParetoOptMDQNAgent(BaseAgent):
    def __init__(self, file_path, pruning_eps, **kwargs):
        super().__init__(**kwargs)

        self.critic = MDQNCritic.load(file_path)
        if self.critic.eps != pruning_eps:
            warnings.warn('Pruning eps of the trained MDQN (eps=%.2f) is different from what is provided (eps=%.2f)' %
                          (self.critic.eps, pruning_eps))
        self.actor = ParetoOptimalPolicy(self.critic, eps=pruning_eps)

    def train(self) -> dict:
        pass

    def add_to_replay_buffer(self, paths):
        pass

    def sample(self, batch_size):
        pass

    def save(self, path):
        pass

