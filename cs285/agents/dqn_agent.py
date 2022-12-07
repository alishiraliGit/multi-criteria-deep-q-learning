import numpy as np

from cs285.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer, PiecewiseSchedule
from cs285.agents.base_agent import BaseAgent
from cs285.policies.argmax_policy import ArgMaxPolicy, PrunedArgMaxPolicy
from cs285.critics.dqn_critic import DQNCritic, PrunedDQNCritic


class DQNAgent(object):
    def __init__(self, env, agent_params):

        self.env = env
        self.agent_params = agent_params

        # Env params
        self.last_obs = self.env.reset()
        self.num_actions = agent_params['ac_dim']

        # Learning params
        self.batch_size = agent_params['batch_size']
        self.learning_starts = agent_params['learning_starts']
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']
        self.exploration = agent_params['exploration_schedule']
        self.optimizer_spec = agent_params['optimizer_spec']

        # Pruning
        prune = True if 'action_pruner' in agent_params else False

        # Actor/Critic
        if prune:
            self.critic = PrunedDQNCritic(agent_params, self.optimizer_spec, agent_params['action_pruner'])
            self.actor = PrunedArgMaxPolicy(self.critic, agent_params['action_pruner'])
        else:
            self.critic = DQNCritic(agent_params, self.optimizer_spec)
            self.actor = ArgMaxPolicy(self.critic)

        # Replay buffer
        lander = agent_params['env_name'].startswith('LunarLander')
        self.replay_buffer = MemoryOptimizedReplayBuffer(
            agent_params['replay_buffer_size'], agent_params['frame_history_len'], lander=lander)
        self.replay_buffer_idx = None

        # Counters
        self.t = 0
        self.num_param_updates = 0

    def add_to_replay_buffer(self, paths):
        # step_env() has already added the transition to the replay buffer
        pass

    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """        

        # Store the latest observation ('frame') into the replay buffer
        # The replay buffer used here is 'MemoryOptimizedReplayBuffer' in dqn_utils.py
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        eps = self.exploration.value(self.t)

        # Use epsilon greedy exploration when selecting action
        perform_random_action = (np.random.random() < eps) or (self.t < self.learning_starts)
        if perform_random_action:
            action = self.env.action_space.sample()
        else:
            # HINT: Your actor will take in multiple previous observations ('frames') in order
            # to deal with the partial observability of the environment. Get the most recent 
            # 'frame_history_len' observations using functionality from the replay buffer,
            # and then use those observations as input to your actor. 
            frames = self.replay_buffer.encode_recent_observation()
            action = self.actor.get_action(frames)[0]
        
        # Take a step in the environment using the action from the policy
        self.last_obs, reward, done, _info = self.env.step(action)

        # Store the result of taking this action into the replay buffer
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        # If taking this step resulted in done, reset the env (and the latest observation)
        if done:
            self.last_obs = self.env.reset()

            if self.agent_params['env_name'].startswith('CartPole'):
                self.last_obs = self.last_obs[0]

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [], [], [], [], []

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}

        if self.t > self.learning_starts \
           and self.t % self.learning_freq == 0 \
           and self.replay_buffer.can_sample(self.batch_size):

            log = self.critic.update(
                ob_no, ac_na, next_ob_no, re_n, terminal_n
            )

            # Update the target network periodically
            if self.num_param_updates % self.target_update_freq == 0:
                self.critic.update_target_network()

            self.num_param_updates += 1

        self.t += 1

        return log


class LoadedDQNAgent(BaseAgent):
    def __init__(self, file_path, **kwargs):
        super().__init__(**kwargs)

        self.critic = DQNCritic.load(file_path)
        self.actor = ArgMaxPolicy(self.critic)

    def train(self) -> dict:
        pass

    def add_to_replay_buffer(self, paths):
        pass

    def sample(self, batch_size):
        pass

    def save(self, path):
        pass
