"""
Plan of attack:

1) Binned plot of trajectory mortality (could likely use rtg to get this) and q-values assigned by policy
- Find a way to load the final prunedCQL critics (see 2nd stage training scripts)
- Load the eval_paths data
- Use critics to get Q-values for each eval_path
- Extract RTG which gives mortality rate 
- Make binned plot

2) Compare Q-value distribution of survivor and non-survivor trajectories
- Split Q-value and RTG dataset by mortality
- Create histogram of Q-values, coloring distribution of each variable differently

3) Compare mortality rate in trajectories with dropped state-action pair vs. non-dropped
- Load all eval_paths, see whether these could also include the traj_id
- Load (best a list of) dropped state-action pairs
- For each path determine whether state-action pair has been dropped or not
- For each path determine whether patient died at the end of the trajectory
- Create df with best traj_id, dropped?, mortality?
- Compute avg. mortality by dropping decision

[If 3 seems useful i.e. we are dropping actions 
that lead to higher mortality then this metric could later be used to tune the pruning eps]

# Get the Q-values
        qa_values_na = critic.qa_values(obs_n)

        qa_values_na = ptu.from_numpy(qa_values_na)
        ac_n = ptu.from_numpy(ac_n).to(torch.long)

        q_values_n = torch.gather(qa_values_na, 1, ac_n.unsqueeze(1)).squeeze(1)
        q_values_n = ptu.to_numpy(q_values_n)

        # Get reward-to-go and transform to dummy indicator of survival
        rtg_n = utils.discounted_cumsum(re_n, params['gamma']) / 100  # to make this a mortality indicator
        rtg_n = (rtg_n + 1) / 2

        # Get values to append for plot
        mean_q = np.mean(q_values_n)
        max_rtg = np.max(rtg_n)

"""

import pickle
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import argparse
import glob

import rlcodebase.envs.mimic_utils

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from rlcodebase.infrastructure.utils import pytorch_utils as ptu
from rlcodebase.critics.dqn_critic import DQNCritic, MDQNCritic, ExtendedMDQNCritic, PrunedDQNCritic
from rlcodebase.critics.cql_critic import CQLCritic, PrunedCQLCritic


def get_q_vals(ac_n,re_n,obs_n,critic):
    # Get the Q-values
    qa_values_na = critic.qa_values(obs_n)

    qa_values_na = ptu.from_numpy(qa_values_na)
    ac_n = ptu.from_numpy(ac_n).to(torch.long)

    q_values_n = torch.gather(qa_values_na, 1, ac_n.unsqueeze(1)).squeeze(1)
    q_values_n = ptu.to_numpy(q_values_n)

    # Get reward-to-go and transform to dummy indicator of survival
    rtg_n = utils.discounted_cumsum(re_n, params['gamma']) / 100  # to make this a mortality indicator
    rtg_n = (rtg_n + 1) / 2

    # Get values to append for plot
    mean_q = np.mean(q_values_n)
    max_rtg = np.max(rtg_n)

    return q_values_n, rtg_n, mean_q, max_rtg

def preprocess_bins(traj_info,q_label='mean_q', baseline = False):
    #bins = np.linspace(np.min(traj_info['mean_q']), np.max(traj_info['mean_q']), 100)
    #bins = np.linspace(np.min(traj_info[q_label])*1.1, np.max(traj_info[q_label])*0.9, 50)
    bins = np.linspace(np.min(traj_info[q_label])*1.1, np.max(traj_info[q_label])*0.9, 20)


    group = traj_info.groupby(pd.cut(traj_info.mean_q, bins))

    plot_centers = (bins[:-1] + bins[1:]) / 2

    plot_values = group.survive.mean().fillna(0)
    plot_ci = (group.survive.std() / np.sqrt(group.survive.count())).fillna(0)

    plot_ci = 1.96*np.sqrt((group.survive.mean()*(1-group.survive.mean()))/ group.survive.count()).fillna(0)
    plot_range = group.survive.std().fillna(0)

    indices = group.survive.count() > 20

    if baseline:
        plot_values = group.survive_b.mean().fillna(0)
        plot_ci = (group.survive_b.std() / np.sqrt(group.survive_b.count())).fillna(0)

        plot_ci = 1.96*np.sqrt((group.survive_b.mean()*(1-group.survive_b.mean()))/ group.survive_b.count()).fillna(0)
        plot_range = group.survive_b.std().fillna(0)

        indices = group.survive_b.count() > 20

    plot_centers = plot_centers[indices]
    plot_values = plot_values[indices]
    plot_ci = plot_ci[indices]
    plot_range = plot_range[indices]

    return plot_centers, plot_values, plot_ci, plot_range



def plot_binned_mortality(eval_paths, params, critic, critic_baseline=None, folder_path="test"):

    fig_path_ = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'figs')

    all_q_values = []
    all_rtgs = []

    mean_q_per_traj = []
    max_rtg_per_traj = []

    all_q_values_b = []
    all_rtgs_b = []

    mean_q_per_traj_b = []
    max_rtg_per_traj_b = []

    for eval_path in eval_paths:
        obs_n = eval_path['observation']
        if obs_n.ndim == 1:
            obs_n = obs_n[:, np.newaxis]

        ac_n = eval_path['action']
        re_n = eval_path['sparse_90d_rew']

        q_values_n, rtg_n, mean_q, max_rtg = get_q_vals(ac_n,re_n,obs_n,critic)

        # Get the Q-values
        qa_values_na = critic.qa_values(obs_n)

        qa_values_na = ptu.from_numpy(qa_values_na)
        ac_n = ptu.from_numpy(ac_n).to(torch.long)

        q_values_n = torch.gather(qa_values_na, 1, ac_n.unsqueeze(1)).squeeze(1)
        q_values_n = ptu.to_numpy(q_values_n)

        # Get reward-to-go and transform to dummy indicator of survival
        rtg_n = utils.discounted_cumsum(re_n, params['gamma']) / 100  # to make this a mortality indicator
        rtg_n = (rtg_n + 1) / 2

        # Get values to append for plot
        mean_q = np.mean(q_values_n)
        max_rtg = np.max(rtg_n)

        mean_q_per_traj.append(mean_q)
        max_rtg_per_traj.append(max_rtg)

        # Append
        all_q_values.append(q_values_n)
        all_rtgs.append(rtg_n)

        if critic_baseline != None:
            q_values_n_b, rtg_n_b, mean_q_b, max_rtg_b = get_q_vals(ac_n,re_n,obs_n,critic_baseline)
            mean_q_per_traj_b.append(mean_q_b)
            max_rtg_per_traj_b.append(max_rtg_b)

            # Append
            all_q_values_b.append(q_values_n_b)
            all_rtgs_b.append(rtg_n_b)

    all_q_values = np.concatenate(all_q_values, axis=0)
    all_rtgs = np.concatenate(all_rtgs, axis=0)

    traj_info = pd.DataFrame()
    traj_info['mean_q'] = all_q_values  # mean_q_per_traj
    traj_info['survive'] = all_rtgs  # max_rtg_per_traj

    if critic_baseline != None:
        all_q_values_b = np.concatenate(all_q_values_b, axis=0)
        all_rtgs_b = np.concatenate(all_rtgs_b, axis=0)
        traj_info['mean_q_b'] = all_q_values_b  # mean_q_per_traj
        traj_info['survive_b'] = all_rtgs_b  # max_rtg_per_traj


    rho = np.corrcoef(all_rtgs, all_q_values)[0, 1]
    print(rho)

    print(traj_info.describe())

    folder_path_split = folder_path.split('_')
    method = folder_path_split[0]
    eps = folder_path_split[1]

    # create the binned plot

    #bins = np.linspace(np.min(traj_info['mean_q']), np.max(traj_info['mean_q']), 100)
    bins = np.linspace(np.min(traj_info['mean_q'])*1.1, np.max(traj_info['mean_q'])*0.9, 50)


    plot_centers, plot_values, plot_ci, plot_range = preprocess_bins(traj_info, 'mean_q', False)

    if critic_baseline != None:
        plot_centers_b, plot_values_b, plot_ci_b, plot_range_b = preprocess_bins(traj_info, 'mean_q_b', True)
    
    """
    # create the binned plot
    plt.figure()

    bins = np.linspace(np.maximum(np.min(traj_info['mean_q']), 0), np.max(traj_info['mean_q']), 25)
    group = traj_info.groupby(pd.cut(traj_info.mean_q, bins))

    plot_centers = (bins[:-1] + bins[1:]) / 2
    plot_values = group.survive.mean().fillna(0)
    plot_ci = (group.survive.std() / np.sqrt(group.survive.count())).fillna(0)

    plot_ci = 1.96*np.sqrt((group.survive.mean()*(1-group.survive.mean()))/ group.survive.count()).fillna(0)
    plot_range = group.survive.std().fillna(0)

    indices = group.survive.count() > 10

    plot_centers = plot_centers[indices]
    plot_values = plot_values[indices]
    plot_ci = plot_ci[indices]
    plot_range = plot_range[indices]
    

    plt.plot(plot_centers, plot_values, label='MCQL alpha=10') #f'{method} eps={eps}'

    # 1 std around mean per bin
    #plt.fill_between(plot_centers, plot_values - plot_range, plot_values + plot_range, color="b", alpha=0.2)
    # binomial confidence interval
    plt.fill_between(plot_centers, plot_values - plot_ci, plot_values + plot_ci, color="g", alpha=0.2)

    if critic_baseline != None:
        plt.plot(plot_centers_b, plot_values_b, label=f'baseline model')
        plt.fill_between(plot_centers_b, plot_values_b - plot_ci_b, plot_values_b + plot_ci_b, color="grey", alpha=0.2)

    plt.ylim(0, 1)
    
    plt.ylabel('Survival rate')
    plt.xlabel('Mean Q-value per trajectory')
    #plt.title(f'Survival rate by Q-value {method} eps={eps}')

    plt.legend(loc='best')
    plt.tight_layout()

    if params['save']:
        plt.savefig(os.path.join(fig_path_, folder_path + '_survival_by_Q.pdf'))
    """

    plt.plot(plot_centers, plot_values, color="g", label='MCQL alpha=10') #f'{method} eps={eps}'

    # 1 std around mean per bin
    #plt.fill_between(plot_centers, plot_values - plot_range, plot_values + plot_range, color="b", alpha=0.2)
    plt.fill_between(plot_centers, plot_values - plot_ci, plot_values + plot_ci, color="g", alpha=0.2)

    if critic_baseline != None:
        plt.plot(plot_centers_b, plot_values_b, label=f'baseline model')
        plt.fill_between(plot_centers_b, plot_values_b - plot_ci_b, plot_values_b + plot_ci_b, color="grey", alpha=0.2)

    plt.ylabel('Survival rate')
    plt.xlabel('Mean Q-value per trajectory')
    #plt.title(f'Survival rate by Q-value {folder_path}')

    plt.legend(loc='best')
    plt.tight_layout()

    if params['save']:
        plt.savefig(os.path.join(fig_path_, folder_path + '_survival_by_Q.pdf'))

    plt.show()

    ###############################
    ####### Plot histogram ########
    ###############################

    survived_hist_data = traj_info[traj_info['survive'] == 1]
    nonsurvived_hist_data = traj_info[traj_info['survive'] == 0]

    survived_hist_data['mean_q']

    #bins = np.linspace(20, 80, 50)

    bins = np.linspace(min(traj_info['mean_q']), max(traj_info['mean_q']), 40)

    plt.hist(survived_hist_data['mean_q'], bins, density=True, alpha=0.5, label='survivors')
    plt.hist(nonsurvived_hist_data['mean_q'], bins, density=True, alpha=0.5, label='nonsurvivors')
    plt.ylabel('Probability')
    plt.xlabel('Mean Q-value per trajectory')

    plt.title(f'Q-value historgram {folder_path}')
    plt.legend(loc='upper right')

    if params['save']:
        plt.savefig(os.path.join(fig_path_, folder_path + '_Q_val_hist.pdf'))

    plt.show()


if __name__ == "__main__":

    ###############################
    ####### Parse arguments #######
    ###############################

    parser = argparse.ArgumentParser()

    # Env
    parser.add_argument('--env_name', type=str, default='LunarLander-Customizable')
    parser.add_argument('--env_rew_weights', type=float, nargs='*', default='None')
    parser.add_argument('--gamma', type=float, nargs='*', default=1)

    # get prefix information for models
    parser.add_argument('--critic_prefix', type=str, default="pCQLv2")  # the model I want to load
    # parser.add_argument('--pruning_file_prefix', type=str, default="MIMICCQL") #the models based on which we prune

    # Pruned models loaded
    parser.add_argument('--pruned', action='store_true',
                        help='Specify whether the loaded models are trained after pareto-pruning')

    # MDQN
    parser.add_argument('--mdqn', action='store_true')

    # EMDQN
    parser.add_argument('--emdqn', action='store_true')

    # System
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)

    # offline RL?
    parser.add_argument('--offline', action='store_true')
    # Giving a default here since we mainly want to run this script for evaluation of the MIMIC data 
    parser.add_argument('--buffer_path', type=str, default='./Replay_buffer_extraction/Encoded_paths3_all_rewards.pkl')

    # CQL
    parser.add_argument('--cql', action='store_true')
    parser.add_argument('--cql_alpha', type=float, default=0.2, help='Higher values indicated stronger OOD penalty.')

    # specify learning curves to display
    parser.add_argument('--opt_action_tag', type=str, default="opt_actions")
    parser.add_argument('--pareto_action_tag', type=str, default="pruned_actions")

    # Check whether plot should be saved
    parser.add_argument('--save', action='store_true')

    # add baseline model if needed
    parser.add_argument('--baseline_model', default=None)  # should be prefix of model

    args = parser.parse_args()

    # Convert to dictionary
    params = vars(args)

    mdqn = params['mdqn']
    cql = params['cql']
    emdqn = params['emdqn']

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    ##################################
    # Load critics
    ##################################
    # set device
    ptu.init_gpu(
        use_gpu=not params['no_gpu'],
        gpu_id=params['which_gpu']
    )

    if params['pruned']:

        if mdqn or emdqn:
            if mdqn:
                pruning_folder_paths = glob.glob(os.path.join(data_path, params['critic_prefix'] + '*'))
                assert len(pruning_folder_paths) == 1
                critic_file_path = os.path.join(pruning_folder_paths[0], 'dqn_agent.pt')
                if mdqn:
                    critics = [MDQNCritic.load(f) for f in critic_file_path]
                else:
                    critics = [ExtendedMDQNCritic.load(f) for f in critic_file_path]

        elif cql:
            pruning_folder_paths = glob.glob(os.path.join(data_path, params['critic_prefix'] + '*'))
            critic_file_path = [os.path.join(f, 'dqn_agent.pt') for f in pruning_folder_paths]
            critics = [PrunedCQLCritic.load(f) for f in critic_file_path]
        else:
            pruning_folder_paths = glob.glob(os.path.join(data_path, params['critic_prefix'] + '*'))
            critic_file_path = [os.path.join(f, 'dqn_agent.pt') for f in pruning_folder_paths]
            critics = [PrunedDQNCritic.load(f) for f in critic_file_path]
    else:
        if cql:
            pruning_folder_paths = glob.glob(os.path.join(data_path, params['critic_prefix'] + '*'))
            critic_file_path = [os.path.join(f, 'dqn_agent.pt') for f in pruning_folder_paths]
            critics = [CQLCritic.load(f) for f in critic_file_path]
        else:
            pruning_folder_paths = glob.glob(os.path.join(data_path, params['critic_prefix'] + '*'))
            critic_file_path = [os.path.join(f, 'dqn_agent.pt') for f in pruning_folder_paths]
            critics = [DQNCritic.load(f) for f in critic_file_path]

    # Adding result of baseline sparse DQN if needed
    if params['baseline_model'] is not None:
        prefix_b = params['baseline_model']
        folder_paths_b = glob.glob(os.path.join(data_path, prefix_b + '*'))
        critic_file_path_b = [os.path.join(f, 'dqn_agent.pt') for f in folder_paths_b]
        if cql:
            critics_b = [CQLCritic.load(f) for f in critic_file_path_b]
        else:
            critics_b = [DQNCritic.load(f) for f in critic_file_path_b]

        #critic_file_path = critic_file_path + critic_file_path_b
        #critics = critics + critics_b
    else:
        critics_b = [None]

        # file_paths_b = [glob.glob(os.path.join(f, 'events*'))[0] for f in folder_paths_b]
        # file_paths_ = file_paths_ + file_paths_b
        # folder_paths_ = folder_paths_ + folder_paths_b

    ##################################
    # Load the buffer path data
    ##################################

    # Load the path data
    with open(params['buffer_path'], 'rb') as f:
        all_paths = pickle.load(f)

    if params['env_name'] == 'MIMIC':
        all_paths = rlcodebase.envs.mimic_utils.format_reward(all_paths, weights=params['env_rew_weights'])
    if params['env_name'] == 'MIMIC-Continuous':
        all_paths = rlcodebase.envs.mimic_utils.format_reward(all_paths, weights=params['env_rew_weights'], continuous=True)
    else:
        raise Exception('Invalid env_name!')

    # Let's use 5% as validation set and 15% as hold-out set
    #paths, test_paths = train_test_split(all_paths, test_size=0.15, random_state=params['seed'])

    paths, test_paths = train_test_split(all_paths, test_size=0.05, random_state=params['seed'])

    critic_file_name = [path.split(os.sep)[-2] for path in critic_file_path]
    print(critic_file_name)

    for critic, name in zip(critics, critic_file_name):
        plot_binned_mortality(test_paths, params, critic, critics_b[0], name)
