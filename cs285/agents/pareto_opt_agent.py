import warnings

from cs285.agents.base_agent import BaseAgent
from cs285.policies.pareto_opt_policy import ParetoOptimalPolicy
from cs285.critics.pareto_opt_dqn_critic import ParetoOptCQLCritic


class LoadedParetoOptCQLAgent(BaseAgent):
    def __init__(self, file_paths, pruning_eps, **kwargs):
        super().__init__(**kwargs)

        self.critic = ParetoOptCQLCritic(saved_dqn_critics_paths=file_paths)
        self.actor = ParetoOptimalPolicy(self.critic, eps=pruning_eps)
