import os
import sys
import pickle

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.stats import spearmanr


sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))


def plot_avg_sparse_rew_per_bin(sparse_rew, signal, n_bin, cl='k'):
    bins = np.linspace(np.min(signal), np.max(signal), n_bin + 1)

    avgs = []
    stds = []
    for i_bin in range(n_bin):
        ind = np.logical_and(signal >= bins[i_bin], signal < bins[i_bin + 1])
        if np.any(ind):
            avgs.append(np.mean(sparse_rew[ind]))
            stds.append(np.std(sparse_rew[ind]))
        else:
            avgs.append(np.nan)
            stds.append(np.nan)

    avgs = np.array(avgs)
    stds = np.array(stds)

    mid_bins = (bins[1:] + bins[:-1])/2
    plt.plot(mid_bins, avgs, color=cl)
    plt.fill_between(mid_bins, avgs + stds, avgs - stds, color=cl, alpha=0.1)


if __name__ == '__main__':
    file_path = os.path.join('..', '..', 'Replay_buffer_extraction')
    file_name = 'Encoded_paths13_all_rewards_rtgadjusted'

    sparse_reward_tag = 'sparse_90d_rew'
    gamma = 1

    # Read data
    with open(os.path.join(file_path, file_name + '.pkl'), 'rb') as f:
        paths = pickle.load(f)

    # Concat all paths
    concat_rews = {}
    for path in tqdm(paths):
        for key, val in path.items():
            if (not key.startswith('Reward')) and (not key.startswith(sparse_reward_tag)):
                continue

            if key in concat_rews:
                concat_rews[key] = np.concatenate((concat_rews[key], utils.discounted_cumsum(val, gamma)))
            else:
                concat_rews[key] = utils.discounted_cumsum(val, gamma)

    # Calc rho
    for key, val in concat_rews.items():
        print('mean and std of rtg of %s is %g and %g' % (key, np.mean(val), np.std(val)))
        rho = spearmanr(val, concat_rews[sparse_reward_tag]).correlation
        print('rho of %s and %s is %g\n' % (key, sparse_reward_tag, rho))

    #
    for key, val in concat_rews.items():
        if key is sparse_reward_tag:
            continue

        plt.figure()
        plt.subplot(1, 2, 1)
        plot_avg_sparse_rew_per_bin(concat_rews[sparse_reward_tag], val, 10)
        plt.subplot(1, 2, 2)
        plt.hist(val, 100)
        plt.title(key)

    plt.show()
