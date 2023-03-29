import numpy as np
import pickle

from sklearn.model_selection import train_test_split

from rlcodebase import configs


def get_mimic_dims(paths):
    params = dict()

    params['discrete'] = True

    # Observation and action sizes
    ob_shape = paths[0]['observation'].shape
    ob_dim = 1 if len(ob_shape) == 1 else ob_shape[1]
    ac_dim = 25
    re_shape = paths[0]['reward'].shape
    re_dim = 1 if len(re_shape) == 1 else re_shape[1]
    params['ac_dim'] = ac_dim
    params['ob_dim'] = ob_dim
    params['re_dim'] = re_dim

    return params


def init_mimic_and_update_params(params):
    env = 'MIMIC'

    # Load and format the data
    with open(params['buffer_path'], 'rb') as f:
        all_paths = pickle.load(f)

    all_paths = format_paths(all_paths, params['env_name'], params['env_rew_weights'])

    # Set aside test paths
    train_paths, test_paths = \
        train_test_split(all_paths, test_size=configs.EvalConfig.MIMIC_TEST_SIZE, random_state=params['seed'])

    # Is this env continuous, or discrete? + observation and action sizes
    for k, v in get_mimic_dims(train_paths).items():
        params['agent_params'][k] = v

    return env, train_paths, test_paths


def format_paths(paths, env_name, env_rew_weights=None):
    if env_name == 'MIMIC':
        return format_reward(paths, weights=env_rew_weights)
    elif env_name == 'MIMIC-Continuous':
        return format_reward(paths, weights=env_rew_weights, continuous=True)
    elif env_name == 'MIMIC-MultiInterReward':
        return format_reward(paths, multi_inter=True)
    elif env_name == 'MIMIC-MultiReward':
        return format_reward(paths, multi=True)
    elif env_name == 'MIMIC-MultiContinuousReward':
        return format_reward(paths, multi_continuous=True)
    else:
        raise Exception('Invalid env_name!')


def format_reward(paths, weights=None, multi_inter=False, multi=False, continuous=False, multi_continuous=False):
    """
    Path structure is :
    #   'sparse_90d_rew', 'Reward_matrix_paper',
    #   'Reward_SOFA_1_continous', 'Reward_SOFA_1_binary',
    #   'Reward_SOFA_2_continous', 'Reward_SOFA_2_binary',
    #   'Reward_SOFA_change2_binary', 'Reward_lac_1_continous',
    #   'Reward_lac_1_binary', 'Reward_lac_2_continous', 'Reward_lac_2_binary']

    If multi, or multi_inter, or multi_continuous = True, rewards will be concatenated
    """
    new_paths = []
    for path in paths:
        if multi_inter:
            path['reward'] = np.stack(
                [
                    path['Reward_SOFA_1_continous'],
                    path['Reward_SOFA_1_binary'],
                    path['Reward_SOFA_2_continous'],
                    path['Reward_SOFA_2_binary'],
                    path['Reward_SOFA_change2_binary'],
                    path['Reward_lac_1_continous'],
                    path['Reward_lac_1_binary'],
                    path['Reward_lac_2_continous'],
                    path['Reward_lac_2_binary']
                ],
                axis=1
            )
        elif multi:
            path['reward'] = np.stack(
                [
                    path['sparse_90d_rew_n'],
                    path['Reward_SOFA_1_continous'],
                    path['Reward_SOFA_1_binary'],
                    path['Reward_SOFA_2_continous'],
                    path['Reward_SOFA_2_binary'],
                    path['Reward_SOFA_change2_binary'],
                    path['Reward_lac_1_continous'],
                    path['Reward_lac_1_binary'],
                    path['Reward_lac_2_continous'],
                    path['Reward_lac_2_binary']
                ],
                axis=1
            )
        elif multi_continuous:
            path['reward'] = np.stack(
                [
                    path['sparse_90d_rew_n'],
                    path['Reward_SOFA_1_continous'],
                    path['Reward_SOFA_2_continous'],
                    path['Reward_lac_1_continous'],
                    path['Reward_lac_2_continous'],
                ],
                axis=1
            )
        elif continuous:
            assert len(weights) == 5
            path['reward'] = weights[0] * path['sparse_90d_rew_n'] \
                + weights[1] * path['Reward_SOFA_1_continous'] \
                + weights[2] * path['Reward_SOFA_2_continous'] \
                + weights[3] * path['Reward_lac_1_continous'] \
                + weights[4] * path['Reward_lac_2_continous']
        else:
            assert len(weights) == 10
            path['reward'] = weights[0] * path['sparse_90d_rew_n'] \
                + weights[1] * path['Reward_SOFA_1_continous'] \
                + weights[2] * path['Reward_SOFA_1_binary'] \
                + weights[3] * path['Reward_SOFA_2_continous'] \
                + weights[4] * path['Reward_SOFA_2_binary'] \
                + weights[5] * path['Reward_SOFA_change2_binary'] \
                + weights[6] * path['Reward_lac_1_continous'] \
                + weights[7] * path['Reward_lac_1_binary'] \
                + weights[8] * path['Reward_lac_2_continous'] \
                + weights[9] * path['Reward_lac_2_binary']

        new_paths.append(path)

    return new_paths
