from rlcodebase.envs.gym_utils import empty_wrapper
from rlcodebase.infrastructure.utils import dqn_utils


class EvalConfig:
    MIMIC_TEST_SIZE = 0.05


def get_env_kwargs(env_name):
    if env_name.startswith('LunarLander'):
        kwargs = {
            'optimizer_spec': dqn_utils.lander_optimizer(),
            'q_func': dqn_utils.create_lander_q_network,
            'replay_buffer_size': 50000,
            'batch_size': 32,
            'gamma': 1.00,
            'learning_starts': 1000,
            'learning_freq': 1,
            'frame_history_len': 1,
            'grad_norm_clipping': 10,
            'lander': True,
            'num_timesteps': 500000,  # changed from 300k
            'env_wrappers': empty_wrapper,
            'ep_len': 200,
        }
        kwargs['exploration_schedule'] = dqn_utils.lander_exploration_schedule(kwargs['num_timesteps'])

    elif env_name.startswith('MIMIC'):
        kwargs = {
            'optimizer_spec': dqn_utils.mimic_optimizer(),
            'q_func': dqn_utils.create_mimic_q_network,
            'replay_buffer_size': 500000,
            'batch_size': 32,
            'gamma': 1.00,
            'learning_starts': 0,
            'learning_freq': 1,
            'frame_history_len': 1,
            'grad_norm_clipping': 10,
            'lander': False,
            'num_timesteps': 80001,  # default: 80001
            'env_wrappers': empty_wrapper,
            'exploration_schedule': None,
        }
    else:
        raise NotImplementedError

    return kwargs
