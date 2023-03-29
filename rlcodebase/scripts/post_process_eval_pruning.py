import os

import rlcodebase.envs.mimic_utils

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pickle
import glob
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import argparse
from collections import Counter

from tqdm import tqdm
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from rlcodebase.policies.argmax_policy import PrunedArgMaxPolicy
from rlcodebase.critics.dqn_critic import DQNCritic, MDQNCritic, ExtendedMDQNCritic, PrunedDQNCritic
from rlcodebase.critics.cql_critic import CQLCritic
from rlcodebase.pruners.independent_dqns_pruner import IDQNPruner, ICQLPruner
from rlcodebase.pruners.dqn_pruner import MDQNPruner, ExtendedMDQNPruner
from rlcodebase.infrastructure.utils import pytorch_utils as ptu


def sort_the_list(guide,list_to_be_sorted):
        list_to_be_sorted = [x for _, x in sorted(zip(guide, list_to_be_sorted), key=lambda pair: pair[0])]
        return list_to_be_sorted

# import tensorflow as tf

def tag_in_dict(file,tag):
    # load pickle file
    with open(file, 'rb') as f:
        actions_dict_ = pickle.load(f)
    
    result = tag in actions_dict_.keys()
    return result



def get_action_set_data(file, tag, printing=False, convert_to_list = False):
    
    # load pickle file
    with open(file, 'rb') as f:
        actions_dict_ = pickle.load(f)

    # get actions for tag
    if tag not in actions_dict_.keys():
        print(f'Tag {tag} is not in the loaded file')
    actions = actions_dict_[tag]
    
     # Merge all trajectories
    action_set = []
    for actions_per_path in actions:
        if convert_to_list:
            actions_per_path = list(actions_per_path)
        if printing:
            print(actions_per_path)
            print(type(actions_per_path))
        action_set += actions_per_path
        

    return action_set


def get_pareto_set_sizes(action_set):
    # load pickle file
    pareto_sizes = [len(x) for x in action_set]

    return pareto_sizes


def get_flags_per_traj(file, tag='action_flags'):
    with open(file, 'rb') as f:
        actions_dict_ = pickle.load(f)

    # get flags for each action
    flags_ac = actions_dict_[tag]

    flags_traj = [sum(flags) for flags in flags_ac]
    return flags_traj


def get_mort_per_traj(file, tag='mortality_rtg'):
    with open(file, 'rb') as f:
        actions_dict_ = pickle.load(f)

    # get flags for each action
    mort_ac = actions_dict_[tag]

    mort_traj = [max(flags) for flags in mort_ac]
    return mort_traj


def get_q_per_traj(file, tag='mortality_rtg'):
    with open(file, 'rb') as f:
        actions_dict_ = pickle.load(f)

    # get flags for each action
    q_vals = actions_dict_[tag]

    q_traj = [(sum(q) / len(q)) for q in q_vals]
    return q_traj

def get_non_pareto_actions(pareto_action_sets):
    all_actions = list(range(25))
    pruned_action_sets = []
    for actions_sets in pareto_action_sets:
        pruned_action_set = []
        for pareto_set in tqdm(actions_sets):
            non_pareto_set = [action for action in all_actions if action not in pareto_set]
            pruned_action_set.append(non_pareto_set)
        pruned_action_sets.append(pruned_action_set)
    return pruned_action_sets

def unpack_actions(action_sets):
    save_list = []
    for action_set in action_sets:
        save_list += action_set
    return save_list

def plot_action_dist_by_sofa(phys_actions, pareto_action_set, pruned_action_set, policy_actions, sofa, eps='None',folderpath = "No path", figurepath=os.getcwd()):
    
    """    
    plt.hist(phys_actions, bins=25)
    plt.title("Actions selected by physician")
    plt.xlabel("Action number")
    plt.ylabel("Frequency")
    plt.show()
    """

    print(sofa[:10])
    #filter action files based on SOFA score
    phys_actions_low = [ac for i, ac in enumerate(phys_actions) if sofa[i]<10]
    phys_actions_medium = [ac for i, ac in enumerate(phys_actions) if (sofa[i]<15) and (sofa[i]>=10)]
    phys_actions_high = [ac for i, ac in enumerate(phys_actions) if sofa[i]>=15]

    policy_actions_low = [ac for i, ac in enumerate(policy_actions) if sofa[i]<10]
    policy_actions_medium = [ac for i, ac in enumerate(policy_actions) if (sofa[i]<15) and (sofa[i]>=10)]
    policy_actions_high = [ac for i, ac in enumerate(policy_actions) if sofa[i]>=15]

    pareto_action_set_low = unpack_actions([ac for i, ac in enumerate(pareto_action_set) if sofa[i]<10])
    pareto_action_set_medium = unpack_actions([ac for i, ac in enumerate(pareto_action_set) if (sofa[i]<15) and (sofa[i]>=10)])
    pareto_action_set_high = unpack_actions([ac for i, ac in enumerate(pareto_action_set) if sofa[i]>=15])

    pruned_action_set_low = unpack_actions([ac for i, ac in enumerate(pruned_action_set) if sofa[i]<10])
    pruned_action_set_medium = unpack_actions([ac for i, ac in enumerate(pruned_action_set) if (sofa[i]<15) and (sofa[i]>=10)])
    pruned_action_set_high = unpack_actions([ac for i, ac in enumerate(pruned_action_set) if sofa[i]>=15])

    #pareto_actions_low = [ac for i, ac in enumerate(pareto_actions) if sofa[i]<5]
    #pareto_actions_medium = [ac for i, ac in enumerate(pareto_actions) if (sofa[i]<15) and (sofa[i]>=5)]
    #pareto_actions_high = [ac for i, ac in enumerate(pareto_actions) if sofa[i]>=15]

    #non_pareto_actions_low = [ac for i, ac in enumerate(non_pareto_actions) if sofa[i]<5]
    #non_pareto_actions_medium = [ac for i, ac in enumerate(non_pareto_actions) if (sofa[i]<15) and (sofa[i]>=5)]
    #non_pareto_actions_high = [ac for i, ac in enumerate(non_pareto_actions) if sofa[i]>=15]

    #plot the action dist
    plot_action_dist(phys_actions_low, policy_actions_low, pruned_action_set_low, eps="Low SOFA")
    plot_action_dist(phys_actions_medium, policy_actions_medium, pruned_action_set_medium, eps="Medium SOFA")
    plot_action_dist(phys_actions_high, policy_actions_high, pruned_action_set_high, eps="High SOFA")

def plot_action_dist(phys_actions, pareto_actions, non_pareto_actions, eps='None',folderpath = "No path", figurepath=os.getcwd()):
    
    """    
    plt.hist(phys_actions, bins=25)
    plt.title("Actions selected by physician")
    plt.xlabel("Action number")
    plt.ylabel("Frequency")
    plt.show()
    """

    # Create action map
    inv_action_map = {}
    count = 0
    for i in range(5):
        for j in range(5):
            inv_action_map[count] = [i,j]
            count += 1
    
    #Create action tuple files
    phys_actions_tuple = [None for i in range(len(phys_actions))]
    pareto_actions_tuple = [None for i in range(len(pareto_actions))]
    non_pareto_actions_tuple = [None for i in range(len(non_pareto_actions))]                                          

    for i in range(len(phys_actions)):
        phys_actions_tuple[i] = inv_action_map[phys_actions[i]]

    for i in range(len(pareto_actions)):
        pareto_actions_tuple[i] = inv_action_map[pareto_actions[i]]

    for i in range(len(non_pareto_actions)):
        non_pareto_actions_tuple[i] = inv_action_map[non_pareto_actions[i]]

    #convert to array
    phys_actions_tuple = np.array(phys_actions_tuple)
    pareto_actions_tuple = np.array(pareto_actions_tuple)
    non_pareto_actions_tuple = np.array(non_pareto_actions_tuple)

    #Create 2d histogram data for each group

    phys_actions_iv = phys_actions_tuple[:,0]
    phys_actions_vaso = phys_actions_tuple[:,1]
    hist, x_edges, y_edges = np.histogram2d(phys_actions_iv, phys_actions_vaso, bins=5)

    pareto_actions_iv = pareto_actions_tuple[:,0]
    pareto_actions_vaso = pareto_actions_tuple[:,1]
    hist2, _, _ = np.histogram2d(pareto_actions_iv, pareto_actions_vaso, bins=5)

    non_pareto_actions_iv = non_pareto_actions_tuple[:,0]
    non_pareto_actions_vaso = non_pareto_actions_tuple[:,1]
    hist3, _, _ = np.histogram2d(non_pareto_actions_iv, non_pareto_actions_vaso, bins=5)

    #Relabel edges 
    x_edges = np.arange(-0.5,5)
    y_edges = np.arange(-0.5,5)

    #Plot distributions

    f, (ax1, ax3) = plt.subplots(2, 1, figsize=(4,8))
    ax1.imshow(np.flipud(hist), cmap="Blues",extent=[x_edges[0], x_edges[-1],  y_edges[0],y_edges[-1]])
    #ax2.imshow(np.flipud(hist2), cmap="OrRd", extent=[x_edges[0], x_edges[-1],  y_edges[0],y_edges[-1]])
    ax3.imshow(np.flipud(hist3), cmap="Greens", extent=[x_edges[0], x_edges[-1],  y_edges[0],y_edges[-1]])

    # ax1.grid(color='b', linestyle='-', linewidth=1)
    # ax2.grid(color='r', linestyle='-', linewidth=1)
    # ax3.grid(color='g', linestyle='-', linewidth=1)

    # Major ticks
    ax1.set_xticks(np.arange(0, 5, 1));
    ax1.set_yticks(np.arange(0, 5, 1));
    #ax2.set_xticks(np.arange(0, 5, 1));
    #ax2.set_yticks(np.arange(0, 5, 1));
    ax3.set_xticks(np.arange(0, 5, 1));
    ax3.set_yticks(np.arange(0, 5, 1));

    # Labels for major ticks
    ax1.set_xticklabels(np.arange(0, 5, 1));
    ax1.set_yticklabels(np.arange(0, 5, 1));
    #ax2.set_xticklabels(np.arange(0, 5, 1));
    #ax2.set_yticklabels(np.arange(0, 5, 1));
    ax3.set_xticklabels(np.arange(0, 5, 1));
    ax3.set_yticklabels(np.arange(0, 5, 1));

    # Minor ticks
    ax1.set_xticks(np.arange(-.5, 5, 1), minor=True);
    ax1.set_yticks(np.arange(-.5, 5, 1), minor=True);
    #ax2.set_xticks(np.arange(-.5, 5, 1), minor=True);
    #ax2.set_yticks(np.arange(-.5, 5, 1), minor=True);
    ax3.set_xticks(np.arange(-.5, 5, 1), minor=True);
    ax3.set_yticks(np.arange(-.5, 5, 1), minor=True);

    # Gridlines based on minor ticks
    ax1.grid(which='minor', color='b', linestyle='-', linewidth=1)
    #ax2.grid(which='minor', color='g', linestyle='-', linewidth=1)
    ax3.grid(which='minor', color='r', linestyle='-', linewidth=1)

    im1 = ax1.pcolormesh(x_edges, y_edges, hist, cmap='Blues')
    f.colorbar(im1, ax=ax1, label = "Action counts")

    #im2 = ax2.pcolormesh(x_edges, y_edges, hist2, cmap='Greens')
    #f.colorbar(im2, ax=ax2, label = "Action counts")

    im3 = ax3.pcolormesh(x_edges, y_edges, hist3, cmap='OrRd')
    f.colorbar(im3, ax=ax3, label = "Action counts")

    ax1.set_ylabel('IV fluid dose')
    #ax2.set_ylabel('IV fluid dose')
    ax3.set_ylabel('IV fluid dose')
    ax1.set_xlabel('Vasopressor dose')
    #ax2.set_xlabel('Vasopressor dose')
    ax3.set_xlabel('Vasopressor dose')

    ax1.set_title("Physician policy")
    #ax2.set_title("RL policy actions")
    ax3.set_title("Pruned actions")

    #f.suptitle(f'Action distribution eps = {eps}')
    plt.tight_layout()

    """
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4,12))
    ax1.imshow(np.flipud(hist), cmap="Blues",extent=[x_edges[0], x_edges[-1],  y_edges[0],y_edges[-1]])
    ax2.imshow(np.flipud(hist2), cmap="OrRd", extent=[x_edges[0], x_edges[-1],  y_edges[0],y_edges[-1]])
    ax3.imshow(np.flipud(hist3), cmap="Greens", extent=[x_edges[0], x_edges[-1],  y_edges[0],y_edges[-1]])

    # ax1.grid(color='b', linestyle='-', linewidth=1)
    # ax2.grid(color='r', linestyle='-', linewidth=1)
    # ax3.grid(color='g', linestyle='-', linewidth=1)

    # Major ticks
    ax1.set_xticks(np.arange(0, 5, 1));
    ax1.set_yticks(np.arange(0, 5, 1));
    ax2.set_xticks(np.arange(0, 5, 1));
    ax2.set_yticks(np.arange(0, 5, 1));
    ax3.set_xticks(np.arange(0, 5, 1));
    ax3.set_yticks(np.arange(0, 5, 1));

    # Labels for major ticks
    ax1.set_xticklabels(np.arange(0, 5, 1));
    ax1.set_yticklabels(np.arange(0, 5, 1));
    ax2.set_xticklabels(np.arange(0, 5, 1));
    ax2.set_yticklabels(np.arange(0, 5, 1));
    ax3.set_xticklabels(np.arange(0, 5, 1));
    ax3.set_yticklabels(np.arange(0, 5, 1));

    # Minor ticks
    ax1.set_xticks(np.arange(-.5, 5, 1), minor=True);
    ax1.set_yticks(np.arange(-.5, 5, 1), minor=True);
    ax2.set_xticks(np.arange(-.5, 5, 1), minor=True);
    ax2.set_yticks(np.arange(-.5, 5, 1), minor=True);
    ax3.set_xticks(np.arange(-.5, 5, 1), minor=True);
    ax3.set_yticks(np.arange(-.5, 5, 1), minor=True);

    # Gridlines based on minor ticks
    ax1.grid(which='minor', color='b', linestyle='-', linewidth=1)
    ax2.grid(which='minor', color='g', linestyle='-', linewidth=1)
    ax3.grid(which='minor', color='r', linestyle='-', linewidth=1)

    im1 = ax1.pcolormesh(x_edges, y_edges, hist, cmap='Blues')
    f.colorbar(im1, ax=ax1, label = "Action counts")

    im2 = ax2.pcolormesh(x_edges, y_edges, hist2, cmap='Greens')
    f.colorbar(im2, ax=ax2, label = "Action counts")

    im3 = ax3.pcolormesh(x_edges, y_edges, hist3, cmap='OrRd')
    f.colorbar(im3, ax=ax3, label = "Action counts")

    ax1.set_ylabel('IV fluid dose')
    ax2.set_ylabel('IV fluid dose')
    ax3.set_ylabel('IV fluid dose')
    ax1.set_xlabel('Vasopressor dose')
    ax2.set_xlabel('Vasopressor dose')
    ax3.set_xlabel('Vasopressor dose')

    ax1.set_title("Physician policy")
    ax2.set_title("RL policy actions")
    ax3.set_title("Non pareto-set actions")

    #f.suptitle(f'Action distribution eps = {eps}')
    plt.tight_layout()
    """

    if params['save']:
        plt.savefig(os.path.join(fig_path_, folderpath + '_action_dist_phy_rl_prun.pdf'))

    if params['show']:
        plt.show()

def run_offline_policy_eval( eval_pruner, n_iter=1, eval_policy=None, buffer_path=None, pruning_critic=None, gamma = 1):
    # TODO: Hard-coded
    print_period = 1

    opt_actions = []
    policy_actions = []
    pruned_actions = []
    action_flags = []

    all_q_values = []
    all_rtgs = []

    if buffer_path is not None:
        # Load replay buffer data
        with open(buffer_path, 'rb') as f:
            all_paths = pickle.load(f)
        all_paths = rlcodebase.envs.mimic_utils.format_reward(all_paths, params['env_rew_weights'])
        # Evaluate on 5% hold-out set
        _, paths = train_test_split(all_paths, test_size=0.05, random_state=params['seed'])
        n_iter = 1

    for itr in range(n_iter):
        if itr % print_period == 0:
            print("\n\n********** Iteration %i ************" % itr)

        # Collect trajectories
        # Not implemented for online learning / better eval approach exists as part of training process
        """
        if buffer_path is None:
            paths, envsteps_this_batch = utils.sample_trajectories(
                self.env,
                opt_policy,
                min_timesteps_per_batch=self.params['batch_size'],
                max_path_length=self.params['ep_len'],
                render=False
            )
        """

        # Get optimal actions, pruned_actions and policy actions
        for path in tqdm(paths):
            # get actions
            opt_actions.append(path['action'].astype(int).tolist())

            # get pareto action sets per traj
            pareto_actions = eval_pruner.get_list_of_available_actions(path['observation'])
            pruned_actions.append(pareto_actions)

            # Flag if action not in pareto-set
            flags = [1 if path['action'][i] in pareto_actions[i] else 0 for i in range(len(path['action']))]
            action_flags.append(flags)

            # Get reward to go (and transform to mortality indicator, assummes Gamma == 1)
            rtg_n = utils.discounted_cumsum(path['reward'],
                                            gamma) / 100  # to make this a mortality indicator
            rtg_n = (rtg_n + 1) / 2
            all_rtgs.append(rtg_n)

            # Get Q-values
            if pruning_critic is not None:
                qa_values_na = pruning_critic.qa_values(path['observation'])

                qa_values_na = ptu.from_numpy(qa_values_na)
                ac_n = ptu.from_numpy(path['action']).to(torch.long)
                q_values_n = torch.gather(qa_values_na, 1, ac_n.unsqueeze(1)).squeeze(1)
                q_values_n = list(ptu.to_numpy(q_values_n))
                # print(q_values_n)
                all_q_values.append(q_values_n)
            
            #Get policy recommeded actions
            if eval_policy is not None:
                traj_actions = []
                for ob in path['observation']:
                    ac_n_policy = eval_policy.get_action(ob)
                    traj_actions.append(ac_n_policy)
                policy_actions.append(traj_actions)

    # Log/save
    # Not implemented for now
    """
    with open(os.path.join(self.log_dir, 'actions.pkl'), 'wb') as f:
        if pruning_critic is not None:
            pickle.dump({'opt_actions': opt_actions, 'pruned_actions': pruned_actions, 'action_flags': action_flags,
                            'mortality_rtg': all_rtgs, 'q_vals': all_q_values}, f)
        else:
            pickle.dump({'opt_actions': opt_actions, 'pruned_actions': pruned_actions, 'action_flags': action_flags,
                            'mortality_rtg': all_rtgs}, f)
    """
    output = {'opt_actions': opt_actions, 'pruned_actions': pruned_actions, 'action_flags': action_flags,
                            'mortality_rtg': all_rtgs, 'policy_actions': policy_actions}

    return output

def survival_by_policy_adherence(phys_actions,policy_actions,mortality,eps, print_df=False):

    # Create action map
    inv_action_map = {}
    count = 0
    for i in range(5):
        for j in range(5):
            inv_action_map[count] = [i,j]
            count += 1
    
    #Create action tuple files
    phys_actions_tuple = [None for i in range(len(phys_actions))]
    policy_actions_tuple = [None for i in range(len(policy_actions))]                                          

    for i in range(len(phys_actions)):
        phys_actions_tuple[i] = inv_action_map[phys_actions[i]]

    for i in range(len(policy_actions)):
        policy_actions_tuple[i] = inv_action_map[policy_actions[i]]

    #convert to array
    phys_actions_tuple = np.array(phys_actions_tuple)
    policy_actions_tuple = np.array(policy_actions_tuple)
    mortality = np.array(mortality)

    #Get dosing data
    phys_actions_iv = phys_actions_tuple[:,0]
    phys_actions_vaso = phys_actions_tuple[:,1]

    policy_actions_iv = policy_actions_tuple[:,0]
    policy_actions_vaso = policy_actions_tuple[:,1]

    #Get dosing difference
    iv_difference = phys_actions_iv - policy_actions_iv
    vaso_difference = phys_actions_vaso - policy_actions_vaso

    #Hack: bring data into a dataframe to collapse by dosing difference
    iv_dict = {'iv_difference':iv_difference, 'survival': mortality}
    iv_df = pd.DataFrame(iv_dict)
    iv_df_grouped = iv_df.groupby('iv_difference',as_index=False).mean()
    iv_df_grouped_std = iv_df.groupby('iv_difference',as_index=False).std()

    if print_df:
        print(iv_df_grouped)

    vaso_dict = {'vaso_difference':vaso_difference, 'survival': mortality}
    vaso_df = pd.DataFrame(vaso_dict)
    vaso_df_grouped = vaso_df.groupby('vaso_difference',as_index=False).mean()
    vaso_df_grouped_std = vaso_df.groupby('vaso_difference',as_index=False).std()

    if print_df:
        print(vaso_df_grouped)

    #Create plot 

    plt.plot(iv_df_grouped['iv_difference'],iv_df_grouped['survival'])
    plt.fill_between(iv_df_grouped['iv_difference'], iv_df_grouped['survival'] - iv_df_grouped_std['survival'], iv_df_grouped['survival'] + iv_df_grouped_std['survival'], color="b", alpha=0.2)
    plt.ylabel('90d Survival rate')
    plt.xlabel('Difference to policy recommended intravenous fluid dose')
    plt.title(f'Survival by action overlap with policy eps={eps}')
    plt.ylim(0, 1)
    plt.tight_layout()

    if params['show']:
        plt.show()
    
    plt.plot(vaso_df_grouped['vaso_difference'],vaso_df_grouped['survival'])
    plt.fill_between(vaso_df_grouped['vaso_difference'], vaso_df_grouped['survival'] - vaso_df_grouped_std['survival'], vaso_df_grouped['survival'] + vaso_df_grouped_std['survival'], color="b", alpha=0.2)
    plt.ylabel('90d Survival rate')
    plt.xlabel('Difference to policy recommended vasopressor dose')
    plt.title(f'Survival by action overlap with policy eps={eps}')
    plt.ylim(0, 1)
    plt.tight_layout()

    if params['show']:
        plt.show()

if __name__ == "__main__":

    ###############################
    # Parse arguments
    ###############################

    parser = argparse.ArgumentParser()

    # Get prefix information for models
    parser.add_argument('--prefix', type=str, default="pDQNvdl*_eval")

    # Specify learning curves to display
    parser.add_argument('--opt_tag', type=str, default='opt_actions')
    parser.add_argument('--eval_tag', type=str, default='pruned_actions')

    # Add baseline model if needed
    # parser.add_argument('--baseline_model', default=None) #should be prefix of model

    # Check whether plot should be shown
    parser.add_argument('--show', action='store_true')

    # Check whether plot should be saved
    parser.add_argument('--save', action='store_true')

    # get prefix information for models
    parser.add_argument('--critic_prefix', type=str, default="pCQLv2")  # the model I want to load

    # Pruned models loaded
    parser.add_argument('--pruned', action='store_true',
                        help='Specify whether the loaded models are trained after pareto-pruning')
    parser.add_argument('--pruning_file_prefix', type=str, default=None)
    parser.add_argument('--prune_with_idqn', action='store_true')
    parser.add_argument('--prune_with_icql', action='store_true')
    parser.add_argument('--prune_with_mdqn', action='store_true')
    parser.add_argument('--prune_with_emdqn', action='store_true')
    parser.add_argument('--pruning_n_draw', type=int, default=20, help='Look at random_pruner.')

    #env
    parser.add_argument('--env_rew_weights', type=float, nargs='*', default=None)

    #Offline learning
    parser.add_argument('--buffer_path', type=str, default='./Replay_buffer_extraction/Encoded_paths3_all_rewards.pkl')

    # Model type
    parser.add_argument('--mdqn', action='store_true')
    parser.add_argument('--emdqn', action='store_true')
    parser.add_argument('--cql', action='store_true')

    # System
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)

    args = parser.parse_args()

    # Convert to dictionary
    params = vars(args)

    before_eps = len(params['prefix'][:-6])


    mdqn = params['mdqn']
    cql = params['cql']
    emdqn = params['emdqn']

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    prune_with_idqn = params['prune_with_idqn']
    prune_with_icql = params['prune_with_icql']
    prune_with_mdqn = params['prune_with_mdqn']
    prune_with_emdqn = params['prune_with_emdqn']


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
                    print(critic_file_path)
                    print(type(critic_file_path))
                    if type(critic_file_path) != list:
                        #critics = MDQNCritic.load(critic_file_path)
                        critics = DQNCritic.load(critic_file_path)
                    else:    
                        critics = [MDQNCritic.load(f) for f in critic_file_path]
                else:
                    if type(critic_file_path) != list:
                        critics = ExtendedMDQNCritic.load(critic_file_path)
                    else:    
                        critics = [ExtendedMDQNCritic.load(f) for f in critic_file_path]

        elif cql:
            pruning_folder_paths = glob.glob(os.path.join(data_path, params['critic_prefix'] + '*'))
            critic_file_path = [os.path.join(f, 'dqn_agent.pt') for f in pruning_folder_paths]
            #critics = [PrunedCQLCritic.load(f) for f in critic_file_path]
            critics = [CQLCritic.load(f) for f in critic_file_path]
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
        

    ###############################
    # Load data
    ###############################

    # Path settings
    curr_dir = os.getcwd()

    #data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data')
    #fig_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'figs')

    data_path_ = os.path.join(curr_dir, 'data')
    fig_path_ = os.path.join(curr_dir, 'figs')

    if not (os.path.exists(fig_path_)):
        os.makedirs(fig_path_)

    for f in glob.glob(str(os.path.join(data_path_, params['prefix'] + '*'))):
        print(f)
    
    #print(os.path.join(data_path, params['prefix'] + '*'))

    # Get relevant files
    folder_paths_ = glob.glob(os.path.join(data_path_, params['prefix'] + '*'))
    file_paths_ = [glob.glob(os.path.join(f, 'actions*'))[0] for f in folder_paths_]

    #print(folder_paths_)
    #print(file_paths_)

    # Get action_set for optimal and pruned actions
    opt_action_sets = [get_action_set_data(file, params['opt_tag']) for file in file_paths_]
    pareto_action_sets = [get_action_set_data(file, params['eval_tag']) for file in file_paths_]
    mortality_sets = [get_action_set_data(file, 'mortality_rtg',convert_to_list=True) for file in file_paths_]
    pruned_action_sets = get_non_pareto_actions(pareto_action_sets)

    #get all of the biomarker data
    # ['SOFA', 'SIRS', 'Arterial_lactate', 'Arterial_pH', 'BUN', 'HR', 'DiaBP', 'INR', 'MeanBP', 'RR', 'SpO2',
    # 'SysBP', 'Temp_C', 'GCS', 'mechvent', 'paO2', 'paCO2']

    """
    sofa = [get_action_set_data(file, 'SOFA') for file in file_paths_]
    sirs = [get_action_set_data(file, 'SIRS') for file in file_paths_]
    arterial_lactate = [get_action_set_data(file, 'Arterial_lactate') for file in file_paths_]
    arterial_ph = [get_action_set_data(file, 'Arterial_pH') for file in file_paths_]
    bun = [get_action_set_data(file, 'BUN') for file in file_paths_]
    hr = [get_action_set_data(file, 'HR') for file in file_paths_]
    diabp = [get_action_set_data(file, 'DiaBP') for file in file_paths_]
    inr = [get_action_set_data(file, 'INR') for file in file_paths_]
    meanbp = [get_action_set_data(file, 'MeanBP') for file in file_paths_]
    rr = [get_action_set_data(file, 'RR') for file in file_paths_]
    spo2 = [get_action_set_data(file, 'SpO2') for file in file_paths_]
    sysbp = [get_action_set_data(file, 'SysBP') for file in file_paths_]
    tempc = [get_action_set_data(file, 'Temp_C') for file in file_paths_]
    gcs = [get_action_set_data(file, 'GCS') for file in file_paths_]
    mechvent = [get_action_set_data(file, 'mechvent') for file in file_paths_]
    paO2 = [get_action_set_data(file, 'paO2') for file in file_paths_]
    paCO2 = [get_action_set_data(file, 'paCO2') for file in file_paths_]
    """
    
    #final policy action in dict
    policy_actions_in_dict = [tag_in_dict(file, 'policy_actions') for file in file_paths_]

    if all(policy_actions_in_dict):
        policy_action_sets = [get_action_set_data(file, 'policy_actions') for file in file_paths_]


    # Get mortality indicator and number of flags per trajectory (flags == action not in pareto-set)
    flags_per_traj = [get_flags_per_traj(file, 'action_flags') for file in file_paths_]
    mort_per_traj = [get_mort_per_traj(file, 'mortality_rtg') for file in file_paths_]
    q_per_traj = [get_q_per_traj(file, 'q_vals') for file in file_paths_]

    all_flags = [get_action_set_data(file, 'action_flags') for file in file_paths_]
    all_q_vals = [get_action_set_data(file, 'q_vals') for file in file_paths_]

    print(type(all_q_vals))

    # Get pareto_set sizes
    pareto_set_sizes = [get_pareto_set_sizes(action_set) for action_set in pareto_action_sets]

    # Get eps number
    folder_paths_short = [f.split(os.sep)[-1] for f in folder_paths_]  # used as experiment name
    print(folder_paths_short)
    # TODO
    #eps_list = [int(f.split('_')[0][before_eps:]) for f in folder_paths_short]
    eps_list = [-1 for f in folder_paths_short]

    # Sort these lists by eps
    eps_list_sorted = sorted(eps_list)


    folder_paths_short = [x for _, x in sorted(zip(eps_list, folder_paths_short), key=lambda pair: pair[0])]
    folder_paths_ = [x for _, x in sorted(zip(eps_list, folder_paths_), key=lambda pair: pair[0])]
    file_paths_ = [x for _, x in sorted(zip(eps_list, file_paths_), key=lambda pair: pair[0])]

    pareto_set_sizes = [x for _, x in sorted(zip(eps_list, pareto_set_sizes), key=lambda pair: pair[0])]
    opt_action_sets = [x for _, x in sorted(zip(eps_list, opt_action_sets), key=lambda pair: pair[0])]
    pareto_action_sets = [x for _, x in sorted(zip(eps_list, pareto_action_sets), key=lambda pair: pair[0])]

    pruned_action_sets = [x for _, x in sorted(zip(eps_list, pruned_action_sets), key=lambda pair: pair[0])]
    mortality_sets = [x for _, x in sorted(zip(eps_list, mortality_sets), key=lambda pair: pair[0])]

    if all(policy_actions_in_dict):
        policy_action_sets = [x for _, x in sorted(zip(eps_list, policy_action_sets), key=lambda pair: pair[0])]

    #sort the biomarkers

    """
    sofa = sort_the_list(eps_list,sofa)
    sirs = sort_the_list(eps_list,sirs)
    arterial_lactate = sort_the_list(eps_list,arterial_ph)
    bun = sort_the_list(eps_list,bun)
    hr = sort_the_list(eps_list,hr)
    diabp = sort_the_list(eps_list,diabp)
    inr = sort_the_list(eps_list,inr)
    meanbp = sort_the_list(eps_list,meanbp)
    rr = sort_the_list(eps_list,rr)
    spo2 = sort_the_list(eps_list,spo2)
    sysbp = sort_the_list(eps_list,sysbp)
    tempc = sort_the_list(eps_list,tempc)
    gcs = sort_the_list(eps_list,gcs)
    mechvent = sort_the_list(eps_list,mechvent)
    paO2 = sort_the_list(eps_list,paO2)
    paCO2 = sort_the_list(eps_list,paCO2)
    """

    ##################################
    # Pruning (if requested)
    ##################################

    if params['pruned']:
        pruner = None
        pruning_folder_paths = glob.glob(os.path.join(data_path, params['pruning_file_prefix'] + '*'))

        if prune_with_idqn:
            pruning_file_paths = [os.path.join(f, 'dqn_agent.pt') for f in pruning_folder_paths]
            pruner = [IDQNPruner(pruning_eps=eps, saved_dqn_critics_paths=pruning_file_paths) for eps in eps_list_sorted]
        
        elif prune_with_icql:
            pruning_file_paths = [os.path.join(f, 'dqn_agent.pt') for f in pruning_folder_paths]
            pruner = [ICQLPruner(pruning_eps=eps, saved_dqn_critics_paths=pruning_file_paths) for eps in eps_list_sorted]

        elif prune_with_mdqn:
            assert len(pruning_folder_paths) == 1, 'found %d files!' % len(pruning_folder_paths)
            pruning_file_path = os.path.join(pruning_folder_paths[0], 'dqn_agent.pt')
            pruner = [MDQNPruner(n_draw=params['pruning_n_draw'], file_path=pruning_file_path) for _ in eps_list_sorted]

        elif prune_with_emdqn:
            assert len(pruning_folder_paths) == 1, 'found %d files!' % len(pruning_folder_paths)
            pruning_file_path = os.path.join(pruning_folder_paths[0], 'dqn_agent.pt')
            pruner = [ExtendedMDQNPruner(n_draw=params['pruning_n_draw'], file_path=pruning_file_path) for _ in eps_list_sorted]

        elif params['cql']:
            pruning_file_paths = [os.path.join(f, 'dqn_agent.pt') for f in pruning_folder_paths]
            pruner = [ICQLPruner(file_paths=pruning_file_paths, pruning_eps=eps) for eps in eps_list_sorted]

        params['action_pruner'] = pruner

    ##################################
    # Load policies
    ##################################

    #TODO find a way to load action_pruned

    if params['pruned']:
        if type(critics) == list:
            policies = [PrunedArgMaxPolicy(critic=critics[i],action_pruner=pruner[i]) for i in range(len(critics))]
        else:
            policy = PrunedArgMaxPolicy(critic=critics,action_pruner=pruner[0])
    
    #get actions suggested by policy for test data
    outputs_list = []
    if not all(policy_actions_in_dict):
        for i in range(len(policies)): 
            eval_outputs = run_offline_policy_eval(pruner[i], n_iter=1, eval_policy=policies[i], buffer_path=params['buffer_path'], pruning_critic=critics[i], gamma = 1)
            outputs_list.append(eval_outputs)


    """
    exp_name_ = 'p9_eps0.3_alpha100_eval_LunarLander-Customizable'
    # 'p7_eps0.0-0.0_alpha100_eval_LunarLander-Customizable'

    all_folders_ = glob.glob(os.path.join(data_path, exp_name_ + '*'))
    if len(all_folders_) > 1:
        raise Exception('More than one folder with this exp_name prefix found!')
    if len(all_folders_) == 0:
        raise Exception('No such file found!')

    folder_path_ = all_folders_[0]
    file_path_ = os.path.join(folder_path_, 'actions.pkl')
    

    # Load actions
    with open(file_path_, 'rb') as f:
        actions_dict_ = pickle.load(f)

    opt_actions = actions_dict_['opt_actions']
    pruned_actions = actions_dict_['pruned_actions']

    # Merge all trajectories
    optimal_set = []
    for opt_actions_per_path in opt_actions:
        optimal_set += opt_actions_per_path

    pareto_set = []
    for pareto_opt_actions_per_path in pruned_actions:
        pareto_set += pareto_opt_actions_per_path
    
    # Analyze Pareto action space
    pareto_sizes = [len(x) for x in pareto_set]
    """

    #######################################################################
    #################### Action distribution analysis  ####################
    #######################################################################

    #Check physician actions
    for i in range(len(opt_action_sets)):
        eps = eps_list_sorted[i]
        phys_actions = opt_action_sets[i]
        #phys_actions = unpack_actions(outputs_list[i]['opt_actions'])
        pareto_actions = unpack_actions(pareto_action_sets[i])
        #policy_actions = unpack_actions(outputs_list[i]['policy_actions'])
        policy_actions = policy_action_sets[i]
        non_pareto_actions = unpack_actions(pruned_action_sets[i])
        path_for_plot = folder_paths_short[i]
        plot_action_dist(phys_actions, policy_actions, non_pareto_actions, eps, folderpath=path_for_plot, figurepath = fig_path_)
        #print('Now Pareto-actions')
        #plot_action_dist(phys_actions, pareto_actions, non_pareto_actions, eps, folderpath=path_for_plot, figurepath = fig_path_)

        print(len(phys_actions))
        print(len(policy_actions))
        #print(len(sofa[0]))
        print(len(pareto_action_sets[0]))
        print(len(pareto_actions))
        print(len(non_pareto_actions))  

        #sofa_exp = sofa[i]


        #plot_action_dist_by_sofa(phys_actions, pareto_action_sets[i], pruned_action_sets[i], policy_actions, sofa_exp, eps, folderpath=path_for_plot, figurepath = fig_path_)
    
    #######################################################################
    ################## Mortality by behavior vs. policy  ##################
    #######################################################################

    for i in range(len(opt_action_sets)):
        mortality = mortality_sets[i]
        policy_actions = policy_action_sets[i]
        phys_actions = opt_action_sets[i]
        eps = eps_list_sorted[i]
        survival_by_policy_adherence(phys_actions,policy_actions,mortality,eps, print_df=True)

    """
    # Create action map
    inv_action_map = {}
    count = 0
    for i in range(5):
        for j in range(5):
            inv_action_map[count] = [i,j]
            count += 1
    
    #Create action tuple files
    phys_actions_tuple = [None for i in range(len(phys_actions))]
    policy_actions_tuple = [None for i in range(len(policy_actions))]                                          

    for i in range(len(phys_actions)):
        phys_actions_tuple[i] = inv_action_map[phys_actions[i]]

    for i in range(len(policy_actions)):
        policy_actions_tuple[i] = inv_action_map[policy_actions[i]]

    #convert to array
    phys_actions_tuple = np.array(phys_actions_tuple)
    policy_actions_tuple = np.array(policy_actions_tuple)
    mortality = np.array(mortality)

    #Get dosing data
    phys_actions_iv = phys_actions_tuple[:,0]
    phys_actions_vaso = phys_actions_tuple[:,1]

    policy_actions_iv = policy_actions_tuple[:,0]
    policy_actions_vaso = policy_actions_tuple[:,1]

    #Get dosing difference
    iv_difference = phys_actions_iv - policy_actions_iv
    vaso_difference = phys_actions_vaso - policy_actions_vaso

    #Hack: bring data into a dataframe to collapse by dosing difference
    iv_dict = {'iv_difference':iv_difference, 'survival': mortality}
    iv_df = pd.DataFrame(iv_dict)
    iv_df_grouped = iv_df.groupby('iv_difference',as_index=False).mean()
    iv_df_grouped_std = iv_df.groupby('iv_difference',as_index=False).std()
    print(iv_df_grouped)

    vaso_dict = {'vaso_difference':vaso_difference, 'survival': mortality}
    vaso_df = pd.DataFrame(vaso_dict)
    vaso_df_grouped = vaso_df.groupby('vaso_difference',as_index=False).mean()
    vaso_df_grouped_std = vaso_df.groupby('vaso_difference',as_index=False).std()
    print(vaso_df_grouped)

    #Create plot 

    plt.plot(iv_df_grouped['iv_difference'],iv_df_grouped['survival'])
    plt.fill_between(iv_df_grouped['iv_difference'], iv_df_grouped['survival'] - iv_df_grouped_std['survival'], iv_df_grouped['survival'] + iv_df_grouped_std['survival'], color="b", alpha=0.2)
    plt.ylabel('90d Survival rate')
    plt.xlabel('Difference to policy recommended intravenous fluid dose')
    plt.title(f'Survival by action overlap with policy eps={eps}')
    plt.ylim(0, 1)
    plt.tight_layout()

    if params['show']:
        plt.show()
    
    plt.plot(vaso_df_grouped['vaso_difference'],vaso_df_grouped['survival'])
    plt.fill_between(vaso_df_grouped['vaso_difference'], vaso_df_grouped['survival'] - vaso_df_grouped_std['survival'], vaso_df_grouped['survival'] + vaso_df_grouped_std['survival'], color="b", alpha=0.2)
    plt.ylabel('90d Survival rate')
    plt.xlabel('Difference to policy recommended vasopressor dose')
    plt.title(f'Survival by action overlap with policy eps={eps}')
    plt.ylim(0, 1)
    plt.tight_layout()

    if params['show']:
        plt.show()
    """


    ############################################
    # Plot Pareto Dist (action set size)
    ############################################

    sizes = [Counter(pareto_sizes).keys() for pareto_sizes in pareto_set_sizes]  # equals to list(set(words))
    counts = [Counter(pareto_sizes).values() for pareto_sizes in pareto_set_sizes]  # counts the elements' frequency

    for i in range(len(sizes)):
        plt.figure(figsize=(5, 4))

        plt.bar(sizes[i], counts[i], color='blue')
        plt.ylabel('Number of observations')
        plt.xlabel('Pareto set size (# of actions)')
        plt.title(f'Pareto-set size distribution eps={eps_list_sorted[i]}')
        plt.xlim(0, 25)

        if params['save']:
            #plt.savefig(os.path.join(fig_path, folder_paths_short[i].split('_')[0] + '_counts.pdf'))
            plt.savefig(os.path.join(fig_path_, folder_paths_short[i] + '_counts.jpg'))

        if params['show']:
            plt.show()

    ###############################
    # Some analysis
    ###############################

    # What is the size of the pareto sets?
    mean_sizes = np.array([statistics.mean(pareto_sizes) for pareto_sizes in pareto_set_sizes])
    std_sizes = np.array([statistics.stdev(pareto_sizes) for pareto_sizes in pareto_set_sizes])  # for CI

    print(mean_sizes)

    # plot mean_sizes and std by eps

    plt.figure(figsize=(5, 4))

    plt.plot(eps_list_sorted, mean_sizes, color='blue', label="Mean Pareto Set Size")

    # Crude 95% CI approximation
    plt.fill_between(eps_list_sorted, mean_sizes - std_sizes, mean_sizes + std_sizes, color="b", alpha=0.2)

    # plt.plot(eps_list_sorted, std_sizes, color='orange', label="Pareto Set Size Std")
    plt.ylabel('Mean Pareto Set Size')
    plt.xlabel('Eps value (in %)')
    plt.title(f'Mean Pareto-set size by eps')
    plt.ylim(0, 25)
    # plt.legend(loc='best')
    plt.tight_layout()

    if params['save']:
        #plt.savefig(os.path.join(fig_path, folder_paths_short[i].split('_')[0] + '_mean_set_size.pdf'))
        plt.savefig(os.path.join(fig_path_, folder_paths_short[i] + '_mean_set_size.pdf'))

    if params['show']:
        plt.show()

    # print(f'The average number of pareto-optimal actions in the set is {mean_size}.' +
    #      f' The standard deviation is {std_size}')

    # Does the pareto set contain the optimal action?

    # Assign one if pareto set contains optimal action
    pareto_set_accuracies = [[1 if y in x else 0 for x, y in zip(pareto_set, optimal_set)]
                             for pareto_set, optimal_set in zip(pareto_action_sets, opt_action_sets)]

    pareto_mean_accuracy = np.array(
        [statistics.mean(pareto_set_accuracy) * 100 for pareto_set_accuracy in pareto_set_accuracies])
    pareto_std_accuracy = np.array(
        [statistics.stdev(pareto_set_accuracy) * 100 for pareto_set_accuracy in pareto_set_accuracies])

    print(pareto_mean_accuracy)

    # plot pareto_set accuracy

    plt.figure(figsize=(5, 4))

    plt.plot(eps_list_sorted, pareto_mean_accuracy, color='green', label="Mean Pareto Set Accuracy")
    # Crude 95% CI approximation
    plt.fill_between(eps_list_sorted, pareto_mean_accuracy - pareto_std_accuracy,
                     pareto_mean_accuracy + pareto_std_accuracy, color="g", alpha=0.2)

    plt.ylabel('Pareto set accuracy (in %)')
    plt.xlabel('Eps value (in %)')
    plt.title(f'Mean Pareto-set accuracy by eps')
    plt.ylim(0, 100)
    plt.legend(loc='best')
    plt.tight_layout()

    if params['save']:
        #plt.savefig(os.path.join(fig_path, folder_paths_short[i].split('_')[0] + '_mean_pareto_acc.pdf'))
        plt.savefig(os.path.join(fig_path_, folder_paths_short[i] + '_mean_pareto_acc.pdf'))

    if params['show']:
        plt.show()

    # print(f'Overall {pareto_mean_accuracy} % of the pareto sets contain the action selected by the network trained' +
    # f' using the correct reward function')

    # TODO For now not needed for report if result is not surprising
    ## Does the pareto set contain the optimal action depending on pareto_set size?
    # Create results df to analyze accuracy by pareto-set size
    results_dicts = [{"Pareto Set Size": pareto_sizes, "Includes optimal": pareto_set_accuracy}
                     for pareto_sizes, pareto_set_accuracy in zip(pareto_set_sizes, pareto_set_accuracies)]
    results_dfs = [pd.DataFrame(results_dict) for results_dict in results_dicts]

    results_df_grouped_means = [results_df.groupby('Pareto Set Size').mean() for results_df in results_dfs]
    results_df_grouped_stds = [results_df.groupby('Pareto Set Size').std() for results_df in results_dfs]

    print("Mean and std of pareto set accuracy by pareto-set size")

    i = 0
    for results_df_grouped_mean, results_df_grouped_std in zip(results_df_grouped_means, results_df_grouped_stds):

        plt.figure(figsize=(5, 4))

        plt.bar(results_df_grouped_mean.index, results_df_grouped_mean['Includes optimal'], color='blue')
        plt.ylim(0, 1)
        plt.xlim(0, 25)
        plt.ylabel('Pareto set mean accuracy')
        plt.xlabel('Pareto set size (# of actions)')
        plt.title(f'Pareto-set accuracy by size eps={eps_list_sorted[i]}')

        if params['save']:
            #plt.savefig(os.path.join(fig_path, folder_paths_short[i].split('_')[0] + '_acc_by_size.pdf'))
            plt.savefig(os.path.join(fig_path_, folder_paths_short[i] + '_acc_by_size.pdf'))

        if params['show']:
            plt.show()

        # print(results_df_grouped_mean)
        # print(results_df_grouped_std)
        i += 1

    results_dicts = [{"Number of Flags": flags, "Mortality": mort}
                     for flags, mort in zip(flags_per_traj, mort_per_traj)]
    results_dfs = [pd.DataFrame(results_dict) for results_dict in results_dicts]

    results_df_grouped_means = [results_df.groupby('Number of Flags').mean() for results_df in results_dfs]
    results_df_grouped_stds = [results_df.groupby('Number of Flags').std() for results_df in results_dfs]

    print("Flags and survival rate analysis")

    i = 0
    for results_df_grouped_mean, results_df_grouped_std in zip(results_df_grouped_means, results_df_grouped_stds):

        plt.figure(figsize=(5, 4))

        plt.bar(results_df_grouped_mean.index, results_df_grouped_mean['Mortality'], color='blue')
        # plt.ylim(0, 1)
        # plt.xlim(0, 25)
        plt.ylabel('Survival rate')
        plt.xlabel('Number of Pareto actions in trajectory')
        plt.title(f'Survival rate by number of pareto-actions in traj eps={eps_list_sorted[i]}')

        if params['save']:
            #plt.savefig(os.path.join(fig_path, folder_paths_short[i].split('_')[0] + 'mortality_num_flags.pdf'))
            plt.savefig(os.path.join(fig_path_, folder_paths_short[i] + 'mortality_num_flags.pdf'))

        plt.show()
        # if params['show']:
        # plt.show()

        # print(results_df_grouped_mean)
        # print(results_df_grouped_std)
        i += 1

    results_dicts = [{"Flags": flags, "q_vals": q_val}
                     for flags, q_val in zip(all_flags, all_q_vals)]

    results_dfs = [pd.DataFrame(results_dict) for results_dict in results_dicts]

    flagged_dfs = [results_df[results_df['Flags'] == 1] for results_df in results_dfs]
    non_flagged_dfs = [results_df[results_df['Flags'] == 0] for results_df in results_dfs]

    print("Flags and Q_vals analysis")

    i = 0
    for flagged, non_flagged in zip(flagged_dfs, non_flagged_dfs):

        bins = np.linspace(min(np.min(flagged['q_vals']), np.min(non_flagged['q_vals'])), max(np.max(flagged['q_vals']), np.max(non_flagged['q_vals'])), 50)

        plt.hist(flagged['q_vals'], bins, density=True, alpha=0.5, label='Pareto-set actions')
        plt.hist(non_flagged['q_vals'], bins, density=True, alpha=0.5, label='Non pareto-set actions')
        plt.ylabel('Probability')
        plt.xlabel('Q-value per action')
        plt.title(f'Q-value historgram eps {eps_list_sorted[i]}')
        plt.legend(loc='upper right')

        if params['save']:
            #plt.savefig(os.path.join(fig_path, folder_paths_short[i].split('_')[0] + 'hist_flagged_vs_non.pdf'))
            plt.savefig(os.path.join(fig_path_, folder_paths_short[i] + 'hist_flagged_vs_non.pdf'))

        plt.show()
        i += 1

    results_dicts = [{"Flags_traj": flags, "q_traj": q_val}
                     for flags, q_val in zip(flags_per_traj, q_per_traj)]

    results_dfs = [pd.DataFrame(results_dict) for results_dict in results_dicts]

    flagged_dfs = [results_df[results_df['Flags_traj'] > 0] for results_df in results_dfs]
    non_flagged_dfs = [results_df[results_df['Flags_traj'] == 0] for results_df in results_dfs]

    print("Flags and Q_vals per trajectory analysis")

    i = 0
    bins = np.linspace(0, 100, 50)
    for flagged, non_flagged in zip(flagged_dfs, non_flagged_dfs):

        bins = np.linspace(min(np.min(flagged['q_traj']), np.min(non_flagged['q_traj'])), max(np.max(flagged['q_traj']), np.max(non_flagged['q_traj'])), 50)

        plt.hist(flagged['q_traj'], bins, density=True, alpha=0.5, label='Traj with pareto set actions')
        plt.hist(non_flagged['q_traj'], bins, density=True, alpha=0.5, label='Only non-pareto-set actions in traj')
        plt.ylabel('Probability')
        plt.xlabel('Mean Q-value per trajectory')
        plt.title(f'Mean Q-value per traj historgram eps={eps_list_sorted[i]}')
        plt.legend(loc='upper right')

        if params['save']:
            plt.savefig(os.path.join(fig_path_, folder_paths_short[i].split('_')[0] + 'hist_q_per_traj_flagged.pdf'))
            plt.savefig(os.path.join(fig_path_, folder_paths_short[i] + 'hist_q_per_traj_flagged.pdf'))

        plt.show()
        i += 1

"""
import os
import pickle
import glob
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import argparse
from collections import Counter

import numpy as np

import tensorflow as tf

def get_action_set(file, tag):
    #load pickle file
    with open(file, 'rb') as f:
        actions_dict_ = pickle.load(f)
    
    #get actions for tag
    actions = actions_dict_[tag]

    # Merge all trajectories
    action_set = []
    for actions_per_path in actions:
        action_set += actions_per_path

    return action_set

def get_pareto_set_sizes(action_set):
    #load pickle file
    pareto_sizes = [len(x) for x in action_set]

    return pareto_sizes


if __name__ == "__main__":
    
    ###############################
    ####### Parse arguments #######
    ###############################

    parser = argparse.ArgumentParser()

    #get prefix information for models
    parser.add_argument('--prefix', type=str, default="pCQLvdl*_eval")

    #specify learning curves to display
    parser.add_argument('--opt_action_tag', type=str, default="opt_actions")
    parser.add_argument('--pareto_action_tag', type=str, default="pareto_opt_actions")

    #add baseline model if needed
    #parser.add_argument('--baseline_model', default=None) #should be prefix of model

    #Check whether plot should be saved
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()

    # Convert to dictionary
    params = vars(args)

    ###############################
    ########## Load data ##########
    ###############################
    
    # Path settings
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data')
    fig_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'figs')
    if not(os.path.exists(fig_path)):
        os.makedirs(fig_path)

    #get relevant files
    folder_paths_ = glob.glob(os.path.join(data_path, params['prefix'] + '*'))
    file_paths_ = [glob.glob(os.path.join(f, 'actions*'))[0] for f in folder_paths_]

    #get action_set for optimal and pareto actions
    opt_action_sets = [get_action_set(file, params['opt_action_tag']) for file in file_paths_]
    pareto_action_sets = [get_action_set(file, params['pareto_action_tag']) for file in file_paths_]

    #get pareto_set sizes
    pareto_set_sizes = [get_pareto_set_sizes(action_set) for action_set in pareto_action_sets]

    #get eps number
    folder_paths_short = [f.split("\\")[-1] for f in folder_paths_] #used as experiment name
    eps_list = [int(f.split('_')[0][7:]) for f in folder_paths_short]
    
    #sort these lists by eps
    eps_list_sorted = sorted(eps_list)

    folder_paths_short = [x for _, x in sorted(zip(eps_list, folder_paths_short), key=lambda pair: pair[0])]
    folder_paths_ = [x for _, x in sorted(zip(eps_list, folder_paths_), key=lambda pair: pair[0])]
    file_paths_ = [x for _, x in sorted(zip(eps_list, file_paths_), key=lambda pair: pair[0])]

    pareto_set_sizes = [x for _, x in sorted(zip(eps_list, pareto_set_sizes), key=lambda pair: pair[0])]
    opt_action_sets = [x for _, x in sorted(zip(eps_list, opt_action_sets), key=lambda pair: pair[0])]
    pareto_action_sets = [x for _, x in sorted(zip(eps_list, pareto_action_sets), key=lambda pair: pair[0])]
    
    
    exp_name_ = 'p9_eps0.3_alpha100_eval_LunarLander-Customizable'
    # 'p7_eps0.0-0.0_alpha100_eval_LunarLander-Customizable'

    all_folders_ = glob.glob(os.path.join(data_path, exp_name_ + '*'))
    if len(all_folders_) > 1:
        raise Exception('More than one folder with this exp_name prefix found!')
    if len(all_folders_) == 0:
        raise Exception('No such file found!')

    folder_path_ = all_folders_[0]
    file_path_ = os.path.join(folder_path_, 'actions.pkl')
    

    # Load actions
    with open(file_path_, 'rb') as f:
        actions_dict_ = pickle.load(f)

    opt_actions = actions_dict_['opt_actions']
    pareto_opt_actions = actions_dict_['pareto_opt_actions']

    # Merge all trajectories
    optimal_set = []
    for opt_actions_per_path in opt_actions:
        optimal_set += opt_actions_per_path

    pareto_set = []
    for pareto_opt_actions_per_path in pareto_opt_actions:
        pareto_set += pareto_opt_actions_per_path
    
    # Analyze Pareto action space
    pareto_sizes = [len(x) for x in pareto_set]
    

    ##################################
    ######## Plot Pareto Dist ########
    ##################################

    sizes = [Counter(pareto_sizes).keys() for pareto_sizes in pareto_set_sizes]  # equals to list(set(words))
    counts = [Counter(pareto_sizes).values() for pareto_sizes in pareto_set_sizes]  # counts the elements' frequency

    for i in range(len(sizes)):
        plt.figure(figsize=(5, 4))

        plt.bar(sizes[i], counts[i], color='blue')  
        plt.ylabel('Number of observations')
        plt.xlabel('Pareto set size (# of actions)')
        plt.title(f'Pareto-set size distribution eps={eps_list_sorted[i]}')
        plt.xlim(0, 25)

        if params['save']:
            plt.savefig(os.path.join(fig_path, folder_paths_short[i] + '_counts.jpg'))

        plt.show()

    ###############################
    ######## Some analysis ########
    ###############################

    
    ## What is the size of the pareto sets?
    mean_sizes = np.array([statistics.mean(pareto_sizes) for pareto_sizes in pareto_set_sizes])
    std_sizes = np.array([statistics.stdev(pareto_sizes) for pareto_sizes in pareto_set_sizes]) #for CI

    print(mean_sizes)

    #plot mean_sizes and std by eps

    plt.figure(figsize=(5, 4))

    plt.plot(eps_list_sorted, mean_sizes, color='blue', label="Mean Pareto Set Size")

    # Crude 95% CI approximation
    plt.fill_between(eps_list_sorted, mean_sizes - std_sizes, mean_sizes + std_sizes, color="b", alpha=0.2)
    
    #plt.plot(eps_list_sorted, std_sizes, color='orange', label="Pareto Set Size Std")  
    plt.ylabel('Mean Pareto Set Size')
    plt.xlabel('Eps value (in %)')
    plt.title(f'Mean Pareto-set size by eps')
    plt.ylim(0, 25)
    #plt.legend(loc='best')
    plt.tight_layout()

    if params['save']:
        plt.savefig(os.path.join(fig_path, folder_paths_short[i] + '_mean_set_size.jpg'))

    plt.show()

    #print(f'The average number of pareto-optimal actions in the set is {mean_size}.' +
    #      f' The standard deviation is {std_size}')

    ## Does the pareto set contain the optimal action?

    # Assign one if pareto set contains optimal action (here physician action)
    pareto_set_accuracies = [[1 if y in x else 0 for x, y in zip(pareto_set, optimal_set)] 
                                for pareto_set, optimal_set in zip(pareto_action_sets, opt_action_sets)]

    pareto_mean_accuracy = np.array([statistics.mean(pareto_set_accuracy)*100 for pareto_set_accuracy in pareto_set_accuracies])
    pareto_std_accuracy = np.array([statistics.stdev(pareto_set_accuracy)*100 for pareto_set_accuracy in pareto_set_accuracies])

    print(pareto_mean_accuracy)

    #plot pareto_set accuracy

    plt.figure(figsize=(5, 4))

    plt.plot(eps_list_sorted, pareto_mean_accuracy, color='green', label="Mean Pareto Set Accuracy")
    # Crude 95% CI approximation
    plt.fill_between(eps_list_sorted, pareto_mean_accuracy - pareto_std_accuracy, pareto_mean_accuracy + pareto_std_accuracy, color="g", alpha=0.2)

    plt.ylabel('Pareto set accuracy (in %)')
    plt.xlabel('Eps value (in %)')
    plt.title(f'Mean Pareto-set accuracy by eps')
    plt.ylim(0, 100)
    plt.legend(loc='best')
    plt.tight_layout()

    if params['save']:
        plt.savefig(os.path.join(fig_path, folder_paths_short[i] + '_mean_pareto_acc.jpg'))

    plt.show()

    #print(f'Overall {pareto_mean_accuracy} % of the pareto sets contain the action selected by the network trained' +
          #f' using the correct reward function')
    
    # TODO For now not needed for report if result is not surprising
    ## Does the pareto set contain the optimal action depending on pareto_set size?
    # Create results df to analyze accuracy by pareto-set size
    results_dicts = [{"Pareto Set Size": pareto_sizes, "Includes optimal": pareto_set_accuracy} 
                    for pareto_sizes, pareto_set_accuracy in zip(pareto_set_sizes,pareto_set_accuracies)]
    results_dfs = [pd.DataFrame(results_dict) for results_dict in results_dicts]

    results_df_grouped_means = [results_df.groupby('Pareto Set Size').mean() for results_df in results_dfs]
    results_df_grouped_stds = [results_df.groupby('Pareto Set Size').std() for results_df in results_dfs]

    print("Mean and std of pareto set accuracy by pareto-set size")

    i = 0
    for results_df_grouped_mean, results_df_grouped_std in zip(results_df_grouped_means,results_df_grouped_stds):
    
        plt.figure(figsize=(5, 4))

        plt.bar(results_df_grouped_mean.index, results_df_grouped_mean['Includes optimal'], color='blue')
        plt.ylim(0, 1)
        plt.xlim(0, 25)  
        plt.ylabel('Pareto set mean accuracy')
        plt.xlabel('Pareto set size (# of actions)')
        plt.title(f'Pareto-set accuracy by size eps={eps_list_sorted[i]}')

        if params['save']:
            plt.savefig(os.path.join(fig_path, folder_paths_short[i] + '_acc_by_size.jpg'))

        plt.show()
        
        #print(results_df_grouped_mean)
        #print(results_df_grouped_std)
        i += 1
"""
#git filter-branch --index-filter 'git rm -r --cached --ignore-unmatch state_construction/results' HEAD

#To remove any file with the path prefix example/path/to/something, you can run
#git filter-repo --path example/path/to/something--invert-paths
