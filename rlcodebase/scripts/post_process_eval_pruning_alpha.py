import os
import pickle
import glob
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import argparse
from collections import Counter

import numpy as np


# import tensorflow as tf


def get_action_set_data(file, tag):
    # load pickle file
    with open(file, 'rb') as f:
        actions_dict_ = pickle.load(f)

    # get actions for tag
    actions = actions_dict_[tag]

    print(actions[0])

    # Merge all trajectories
    action_set = []
    for actions_per_path in actions:
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

    args = parser.parse_args()

    # Convert to dictionary
    params = vars(args)

    before_eps = len(params['prefix'][:-6])

    ###############################
    # Load data
    ###############################

    # Path settings
    data_path_ = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data')
    fig_path_ = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'figs')
    if not (os.path.exists(fig_path_)):
        os.makedirs(fig_path_)

    # Get relevant files
    folder_paths_ = glob.glob(os.path.join(data_path_, params['prefix'] + '*'))
    file_paths_ = [glob.glob(os.path.join(f, 'actions*'))[0] for f in folder_paths_]

    # Get action_set for optimal and pruned actions
    opt_action_sets = [get_action_set_data(file, params['opt_tag']) for file in file_paths_]
    pareto_action_sets = [get_action_set_data(file, params['eval_tag']) for file in file_paths_]

    # Get mortality indicator and number of flags per trajectory (flags == action not in pareto-set)
    flags_per_traj = [get_flags_per_traj(file, 'action_flags') for file in file_paths_]
    mort_per_traj = [get_mort_per_traj(file, 'mortality_rtg') for file in file_paths_]
    q_per_traj = [get_q_per_traj(file, 'q_vals') for file in file_paths_]

    all_flags = [get_action_set_data(file, 'action_flags') for file in file_paths_]
    all_q_vals = [get_action_set_data(file, 'q_vals') for file in file_paths_]

    # Get pareto_set sizes
    pareto_set_sizes = [get_pareto_set_sizes(action_set) for action_set in pareto_action_sets]

    # Get eps number
    folder_paths_short = [f.split(os.sep)[-1] for f in folder_paths_]  # used as experiment name
    # TODO
    # eps_list = [int(f.split('_')[0][before_eps:]) for f in folder_paths_short]
    alpha_list = [float(f[f.find('alpha'):].split('_')[0][5:]) for f in folder_paths_short]

    # Sort these lists by eps
    alpha_list_sorted = sorted(alpha_list)

    folder_paths_short = [x for _, x in sorted(zip(alpha_list, folder_paths_short), key=lambda pair: pair[0])]
    folder_paths_ = [x for _, x in sorted(zip(alpha_list, folder_paths_), key=lambda pair: pair[0])]
    file_paths_ = [x for _, x in sorted(zip(alpha_list, file_paths_), key=lambda pair: pair[0])]

    pareto_set_sizes = [x for _, x in sorted(zip(alpha_list, pareto_set_sizes), key=lambda pair: pair[0])]
    opt_action_sets = [x for _, x in sorted(zip(alpha_list, opt_action_sets), key=lambda pair: pair[0])]
    pareto_action_sets = [x for _, x in sorted(zip(alpha_list, pareto_action_sets), key=lambda pair: pair[0])]

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

    ##################################
    # Plot Pareto Dist
    ##################################

    sizes = [Counter(pareto_sizes).keys() for pareto_sizes in pareto_set_sizes]  # equals to list(set(words))
    counts = [Counter(pareto_sizes).values() for pareto_sizes in pareto_set_sizes]  # counts the elements' frequency

    for i in range(len(sizes)):
        plt.figure(figsize=(5, 4))

        plt.bar(sizes[i], counts[i], color='blue')
        plt.ylabel('Number of observations')
        plt.xlabel('Pareto set size (# of actions)')
        plt.title(f'Pareto-set size distribution alpha={alpha_list_sorted[i]}')
        plt.xlim(0, 25)

        if params['save']:
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

    # plot mean_sizes and std by alpha

    plt.figure(figsize=(5, 4))

    plt.plot(alpha_list_sorted, mean_sizes, color='blue', label="Mean Pareto Set Size")

    # Crude 95% CI approximation
    plt.fill_between(alpha_list_sorted, mean_sizes - std_sizes, mean_sizes + std_sizes, color="b", alpha=0.2)

    # plt.plot(alpha_list_sorted, std_sizes, color='orange', label="Pareto Set Size Std")
    plt.ylabel('Mean Pareto Set Size')
    plt.xlabel('alpha value (in %)')
    plt.title(f'Mean Pareto-set size by alpha')
    plt.ylim(0, 25)
    # plt.legend(loc='best')
    plt.tight_layout()

    if params['save']:
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

    plt.plot(alpha_list_sorted, pareto_mean_accuracy, color='green', label="Mean Pareto Set Accuracy")
    # Crude 95% CI approximation
    plt.fill_between(alpha_list_sorted, pareto_mean_accuracy - pareto_std_accuracy,
                     pareto_mean_accuracy + pareto_std_accuracy, color="g", alpha=0.2)

    plt.ylabel('Pareto set accuracy (in %)')
    plt.xlabel('alpha value (in %)')
    plt.title(f'Mean Pareto-set accuracy by alpha')
    plt.ylim(0, 100)
    plt.legend(loc='best')
    plt.tight_layout()

    if params['save']:
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
        plt.title(f'Pareto-set accuracy by size alpha={alpha_list_sorted[i]}')

        if params['save']:
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
        plt.title(f'Survival rate by number of pareto-actions in traj alpha={alpha_list_sorted[i]}')

        if params['save']:
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
    # TODO
    # bins = np.linspace(0, 100, 50)
    bins = 50
    for flagged, non_flagged in zip(flagged_dfs, non_flagged_dfs):

        plt.hist(flagged['q_vals'], bins, density=True, alpha=0.5, label='Pareto-set actions')
        plt.hist(non_flagged['q_vals'], bins, density=True, alpha=0.5, label='Non pareto-set actions')
        plt.ylabel('Probability')
        plt.xlabel('Q-value per action')
        plt.title(f'Q-value histogram alpha {alpha_list_sorted[i]}')
        plt.legend(loc='upper right')

        if params['save']:
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
    # TODO
    # bins = np.linspace(0, 100, 50)
    bins = 50
    for flagged, non_flagged in zip(flagged_dfs, non_flagged_dfs):

        plt.hist(flagged['q_traj'], bins, density=True, alpha=0.5, label='Traj with pareto set actions')
        plt.hist(non_flagged['q_traj'], int(bins/2), density=True, alpha=0.5, label='Only non-pareto-set actions in traj')
        plt.ylabel('Probability')
        plt.xlabel('Mean Q-value per trajectory')
        plt.title(f'Mean Q-value per traj histogram alpha={alpha_list_sorted[i]}')
        plt.legend(loc='upper right')

        if params['save']:
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
    alpha_list = [int(f.split('_')[0][7:]) for f in folder_paths_short]

    #sort these lists by eps
    alpha_list_sorted = sorted(alpha_list)

    folder_paths_short = [x for _, x in sorted(zip(alpha_list, folder_paths_short), key=lambda pair: pair[0])]
    folder_paths_ = [x for _, x in sorted(zip(alpha_list, folder_paths_), key=lambda pair: pair[0])]
    file_paths_ = [x for _, x in sorted(zip(alpha_list, file_paths_), key=lambda pair: pair[0])]

    pareto_set_sizes = [x for _, x in sorted(zip(alpha_list, pareto_set_sizes), key=lambda pair: pair[0])]
    opt_action_sets = [x for _, x in sorted(zip(alpha_list, opt_action_sets), key=lambda pair: pair[0])]
    pareto_action_sets = [x for _, x in sorted(zip(alpha_list, pareto_action_sets), key=lambda pair: pair[0])]


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
        plt.title(f'Pareto-set size distribution eps={alpha_list_sorted[i]}')
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

    plt.plot(alpha_list_sorted, mean_sizes, color='blue', label="Mean Pareto Set Size")

    # Crude 95% CI approximation
    plt.fill_between(alpha_list_sorted, mean_sizes - std_sizes, mean_sizes + std_sizes, color="b", alpha=0.2)

    #plt.plot(alpha_list_sorted, std_sizes, color='orange', label="Pareto Set Size Std")  
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

    plt.plot(alpha_list_sorted, pareto_mean_accuracy, color='green', label="Mean Pareto Set Accuracy")
    # Crude 95% CI approximation
    plt.fill_between(alpha_list_sorted, pareto_mean_accuracy - pareto_std_accuracy, pareto_mean_accuracy + pareto_std_accuracy, color="g", alpha=0.2)

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
        plt.title(f'Pareto-set accuracy by size eps={alpha_list_sorted[i]}')

        if params['save']:
            plt.savefig(os.path.join(fig_path, folder_paths_short[i] + '_acc_by_size.jpg'))

        plt.show()

        #print(results_df_grouped_mean)
        #print(results_df_grouped_std)
        i += 1
"""
