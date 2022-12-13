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
    parser.add_argument('--prefix', type=str, default="pDQNvdl*_eval")

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
    data_path_ = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data')
    fig_path_ = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'figs')
    if not(os.path.exists(fig_path_)):
        os.makedirs(fig_path_)

    #get relevant files
    folder_paths_ = glob.glob(os.path.join(data_path_, params['prefix'] + '*'))
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
    
    """
    exp_name_ = 'pDQN_30_eval_MIMIC_11-12-2022_20-06-30'
    all_folders_ = glob.glob(os.path.join(data_path_, exp_name_ + '*'))
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
    """

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
            plt.savefig(os.path.join(fig_path_, folder_paths_short[i] + '_counts.pdf'))

        plt.show()

    ###############################
    ######## Some analysis ########
    ###############################

    
    ## What is the size of the pareto sets?
    mean_sizes = np.array([statistics.mean(pareto_sizes) for pareto_sizes in pareto_set_sizes])
    std_sizes = np.array([statistics.stdev(pareto_sizes) for pareto_sizes in pareto_set_sizes]) #for CI

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
        plt.savefig(os.path.join(fig_path_, folder_paths_short[i] + '_mean_set_size.pdf'))

    plt.show()

    #print(f'The average number of pareto-optimal actions in the set is {mean_size}.' +
    #      f' The standard deviation is {std_size}')

    ## Does the pareto set contain the optimal action?

    # Assign one if pareto set contains optimal action
    pareto_set_accuracies = [[1 if y in x else 0 for x, y in zip(pareto_set, optimal_set)] 
                                for pareto_set, optimal_set in zip(pareto_action_sets, opt_action_sets)]

    pareto_mean_accuracy = np.array([statistics.mean(pareto_set_accuracy)*100 for pareto_set_accuracy in pareto_set_accuracies])
    pareto_std_accuracy = np.array([statistics.stdev(pareto_set_accuracy)*100 for pareto_set_accuracy in pareto_set_accuracies])

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
        plt.savefig(os.path.join(fig_path_, folder_paths_short[i] + '_mean_pareto_acc.pdf'))

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
            plt.savefig(os.path.join(fig_path_, folder_paths_short[i] + '_acc_by_size.pdf'))

        plt.show()
        
        #print(results_df_grouped_mean)
        #print(results_df_grouped_std)
        i += 1

