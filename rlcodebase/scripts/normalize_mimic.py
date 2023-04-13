import sys
import os
import pickle

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from rlcodebase.infrastructure.utils.rl_utils import discounted_cumsum

if __name__ == '__main__':
    do_save = True

    file_path = os.path.join('Replay_buffer_extraction')
    file_name = 'Paths_all_rewards_best'

    sparse_reward_tag = 'sparse_90d_rew'
    gamma = 1

    # Read data
    with open(os.path.join(file_path, file_name + '.pkl'), 'rb') as f:
        paths = pickle.load(f)

    # Concat all paths
    concat_rews = {}
    for path in tqdm(paths):
        path['Reward_pseudo'] = np.ones((len(path[sparse_reward_tag]),))
        for key, val in path.items():
            if key.startswith('Reward') or (key == sparse_reward_tag):
                if key in concat_rews:
                    concat_rews[key] = np.concatenate((concat_rews[key], discounted_cumsum(val, gamma)))
                else:
                    concat_rews[key] = discounted_cumsum(val, gamma)

    # Calc mean and std
    rew_mean = {}
    rew_std = {}
    for key, val in concat_rews.items():
        rew_mean[key] = np.mean(val)
        rew_std[key] = np.std(val)

        print('%s: [%g, %g]' % (key, rew_mean[key], rew_std[key]))

    # Var 1
    normalized_paths = []
    for path in tqdm(paths):
        normalized_path = {}
        for key, val in path.items():
            if key.startswith('Reward'):
                normalized_path[key] = val / rew_std[key]
            elif key == sparse_reward_tag:
                normalized_path[key] = val
                normalized_path[key + '_n'] = val / rew_std[key]
            else:
                normalized_path[key] = val

        normalized_paths.append(normalized_path)

    # Save results
    if do_save:
        with open(os.path.join(file_path, file_name + '_var1.pkl'), 'wb') as f:
            pickle.dump(normalized_paths, f)
