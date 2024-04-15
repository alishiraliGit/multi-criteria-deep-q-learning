import gym
from gym import register
from gym import wrappers
from gym.envs import registry


###################
# Register gym envs
###################

def register_lunar_lander_env(env_name, file_name):
    if env_name not in registry:
        register(
            id=env_name,
            entry_point='rlcodebase.envs.box2d.%s:LunarLander' % file_name,
            max_episode_steps=1000,
            reward_threshold=200,
        )


def register_sepsis_sim_env(env_name, file_name):
    if env_name not in registry:
        register(
            id=env_name,
            entry_point='rlcodebase.envs.%s:SepsisSimWrapper' % file_name,
        )


def register_custom_envs():
    register_lunar_lander_env('LunarLander-v3', 'lunar_lander_v3')
    register_lunar_lander_env('LunarLander-2211', 'lunar_lander_2211')
    register_lunar_lander_env('LunarLander-2151', 'lunar_lander_2151')
    register_lunar_lander_env('LunarLander-2115', 'lunar_lander_2115')
    register_lunar_lander_env('LunarLander-1251', 'lunar_lander_1251')
    register_lunar_lander_env('LunarLander-1215', 'lunar_lander_1215')
    register_lunar_lander_env('LunarLander-1155', 'lunar_lander_1155')
    register_lunar_lander_env('LunarLander-Sparse', 'lunar_lander_0000')
    register_lunar_lander_env('LunarLander-Customizable', 'lunar_lander_customizable_rew_weights')
    register_lunar_lander_env('LunarLander-MultiReward', 'lunar_lander_multi_rew')
    register_lunar_lander_env('LunarLander-MultiInterReward', 'lunar_lander_multi_inter_rew')
    register_lunar_lander_env('LunarLander-MultiInterRewardNoise', 'lunar_lander_multi_inter_rew_noise')

    register_sepsis_sim_env('SepsisSim-MultiReward', 'sepsissimwrapper_multi_rew')
    register_sepsis_sim_env('SepsisSim-Customizable', 'sepsissimwrapper_customizable_rew_weights')


###################
# Init. gym
###################

def init_gym_and_update_params(params):
    """
    Update
    - ep_len
    - agent_params: discrete, ac_dim, ob_dim, re_dim
    """

    # Make the gym environment
    register_custom_envs()

    env = gym.make(params['env_name'])

    # Set env reward weights
    if (params['env_name'] == 'LunarLander-Customizable' or params['env_name'] == 'SepsisSim-Customizable') \
            and params['env_rew_weights'] is not None:
        env.set_rew_weights(params['env_rew_weights'])
    
    # Set env noise level
    if params['env_name'] == 'LunarLander-MultiInterRewardNoise' and params['env_noise_level'] is not None:
        env.set_noise_level(params['env_noise_level'])

    # Add wrappers
    if 'env_wrappers' in params:
        env = add_wrappers(env, params['env_wrappers'])

    # Set random seed
    env.seed(seed=params['seed'])

    # Update maximum length for episodes
    params['ep_len'] = params['ep_len'] or env.spec.max_episode_steps

    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    params['agent_params']['discrete'] = discrete

    # Are the observations images?
    img = len(env.observation_space.shape) > 2

    # Observation and action sizes
    ob_dim = env.observation_space.shape if img else env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
    re_dim = env.reward_dim
    params['agent_params']['ac_dim'] = ac_dim
    params['agent_params']['ob_dim'] = ob_dim
    params['agent_params']['re_dim'] = re_dim

    return env


###################
# Wrapping gym
###################

def empty_wrapper(env):
    return env


class ReturnWrapper(gym.Wrapper):
    def get_episode_rewards(self):
        return list(self.env.return_queue)


def add_wrappers(env, wrappers_func):
    env = wrappers.RecordEpisodeStatistics(env, deque_size=1000)
    env = ReturnWrapper(env)
    env = wrappers_func(env)

    return env
