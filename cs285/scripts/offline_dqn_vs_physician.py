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
"""

from collections import OrderedDict
import pickle
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import argparse
import glob

from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure.atari_wrappers import ReturnWrapper
from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.critics.dqn_critic import DQNCritic, MDQNCritic, ExtendedMDQNCritic, PrunedDQNCritic
from cs285.critics.cql_critic import CQLCritic, PrunedCQLCritic
#from cs285.agents.dqn_agent import DQNAgent
#from cs285.agents.pareto_opt_agent import LoadedParetoOptDQNAgent, LoadedParetoOptMDQNAgent, LoadedParetoOptCQLAgent, LoadedParetoOptExtendedMDQNAgent
from cs285.infrastructure.dqn_utils import register_custom_envs


def plot_binned_mortality(eval_paths, params, critic, folder_path = "test"):

    fig_path_ = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'figs')

    all_q_values = []
    all_rtgs = []

    mean_q_per_traj = []
    max_rtg_per_traj = []

    for eval_path in eval_paths:
        obs_n = eval_path['observation']
        if obs_n.ndim == 1:
            obs_n = obs_n[:, np.newaxis]

        ac_n = eval_path['action']
        re_n = eval_path['reward']

        # Get the Q-values
        qa_values_na = critic.qa_values(obs_n)

        qa_values_na = ptu.from_numpy(qa_values_na)
        ac_n = ptu.from_numpy(ac_n).to(torch.long)

        q_values_n = torch.gather(qa_values_na, 1, ac_n.unsqueeze(1)).squeeze(1)

        q_values_n = ptu.to_numpy(q_values_n)

        # Get reward-to-go and transform to dummy indicator of survival
        rtg_n = utils.discounted_cumsum(re_n, params['gamma']) /100 #to make this a mortality indicator
        rtg_n = (rtg_n + 1)/2

        # Get values to append for plot
        mean_q = np.mean(q_values_n)
        max_rtg = np.max(rtg_n)

        mean_q_per_traj.append(mean_q)
        max_rtg_per_traj.append(max_rtg)

        # Append
        all_q_values.append(q_values_n)
        all_rtgs.append(rtg_n)

    all_q_values = np.concatenate(all_q_values, axis=0)
    all_rtgs = np.concatenate(all_rtgs, axis=0)

    traj_info = pd.DataFrame()
    traj_info['mean_q'] = mean_q_per_traj
    traj_info['survive'] = max_rtg_per_traj

    rho = np.corrcoef(all_rtgs, all_q_values)[0, 1]
    print(rho)

    print(traj_info.describe())

    #create the binned plot

    bins = np.linspace(25, 60,10)
    group = traj_info.groupby(pd.cut(traj_info.mean_q, bins))

    plot_centers = (bins [:-1] + bins [1:])/2
    plot_values = group.survive.mean().fillna(0)
    plot_range = group.survive.std().fillna(0)

    plt.plot(plot_centers, plot_values)

    # 1 std around mean per bin
    #plt.fill_between(plot_centers, plot_values - plot_range, plot_values + plot_range, color="b", alpha=0.2)

    plt.ylabel('Survival rate')
    plt.xlabel('Mean Q-value per trajectory')
    plt.title(f'Survival rate by Q-value {folder_path}')

    if params['save']:
        plt.savefig(os.path.join(fig_path_, folder_path + '_survival_by_Q.jpg'))

    plt.show()

    ###############################
    ####### Plot histogram ########
    ###############################

    survived_hist_data = traj_info[traj_info['survive']==1]
    nonsurvived_hist_data = traj_info[traj_info['survive']==0]

    bins = np.linspace(20, 70, 50)

    plt.hist(survived_hist_data['mean_q'], bins, density=True, alpha=0.5, label='survivors')
    plt.hist(nonsurvived_hist_data['mean_q'], bins, density=True, alpha=0.5, label='nonsurvivors')
    plt.ylabel('Probability')
    plt.xlabel('Mean Q-value per trajectory')
    plt.title(f'Q-value historgram {folder_path}')
    plt.legend(loc='upper right')

    if params['save']:
        plt.savefig(os.path.join(fig_path_, folder_path + '_Q_val_hist.jpg'))

    plt.show()


if __name__ == "__main__":
    
    ###############################
    ####### Parse arguments #######
    ###############################

    parser = argparse.ArgumentParser()

    #Env
    parser.add_argument('--env_rew_weights', type=float, nargs='*', default='None')
    parser.add_argument('--gamma', type=float, nargs='*', default=1)

    #get prefix information for models
    parser.add_argument('--critic_prefix', type=str, default="pCQLvdl_*[0-9]_M") #the model I want to load
    #parser.add_argument('--pruning_file_prefix', type=str, default="MIMICCQL") #the models based on which we prune

    # Pruned models loaded
    parser.add_argument('--pruned', action='store_true', help='Specify whether the loaded models are trained after pareto-pruning')

    # MDQN
    parser.add_argument('--mdqn', action='store_true')

    # EMDQN
    parser.add_argument('--emdqn', action='store_true')

    # System
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)

    #offline RL?
    parser.add_argument('--offline', action='store_true')
    # Giving a default here since we mainly want to run this script for evaluation of the MIMIC data 
    parser.add_argument('--buffer_path', type=str, default='./Replay_buffer_extraction/Encoded_paths3_all_rewards.pkl')

    # CQL
    parser.add_argument('--cql', action='store_true')
    parser.add_argument('--cql_alpha', type=float, default=0.2,help='Higher values indicated stronger OOD penalty.')

    #specify learning curves to display
    parser.add_argument('--opt_action_tag', type=str, default="opt_actions")
    parser.add_argument('--pareto_action_tag', type=str, default="pareto_opt_actions")

    #Check whether plot should be saved
    parser.add_argument('--save', action='store_true')
    

    #add baseline model if needed
    #parser.add_argument('--baseline_model', default=None) #should be prefix of model

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


    ##################################
    # Load the buffer path data
    ##################################

    # Load the path data
    with open(params['buffer_path'], 'rb') as f:
        all_paths = pickle.load(f)

    all_paths = utils.format_reward(all_paths,params['env_rew_weights'])

    # Let's use 5% as validation set and 15% as hold-out set
    paths, test_paths = train_test_split(all_paths, test_size=0.15, random_state=params['seed'])

    critic_file_name = [path.split("\\")[-2] for path in critic_file_path]
    print(critic_file_name)

    for critic, name in zip(critics, critic_file_name):
        plot_binned_mortality(test_paths,params,critic, name)
