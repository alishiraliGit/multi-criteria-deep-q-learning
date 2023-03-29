import os
import pickle
import glob
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from collections import Counter

if __name__ == "__main__":
    do_save = False

    # Path settings
    data_path_ = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data')
    fig_path_ = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'figs')
    if not (os.path.exists(fig_path_)):
        os.makedirs(fig_path_)

    exp_name_ = 'expvar1clr1-5lr2-4_3_offline_cmdqn_alpha5_cql0.001_r1_tuf11000_tuf28000_eval'
    #exp_name_ = 'sanity_check_cmdqn_eval'

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
    pruned_actions = actions_dict_['pruned_actions']
    if 'policy_actions' in actions_dict_:
        policy_actions = actions_dict_['policy_actions']

    # Merge all trajectories
    optimal_set = []
    for opt_actions_per_path in opt_actions:
        optimal_set += opt_actions_per_path

    pareto_set = []
    for pareto_opt_actions_per_path in pruned_actions:
        pareto_set += pareto_opt_actions_per_path

    if 'policy_actions' in actions_dict_:
        policy_set = []
        for pol_actions_per_path in policy_actions:
            policy_set += pol_actions_per_path

    # Analyze Pareto action space
    pareto_sizes = [len(x) for x in pareto_set]
    mean_size = statistics.mean(pareto_sizes)
    std_size = statistics.stdev(pareto_sizes)

    print(f'The average number of pareto-optimal actions in the set is {mean_size}.' +
          f' The standard deviation is {std_size}')

    sizes = Counter(pareto_sizes).keys()  # equals to list(set(words))
    counts = Counter(pareto_sizes).values()  # counts the elements' frequency

    plt.figure(figsize=(5, 4))

    plt.bar(sizes, counts, color='blue')
    plt.ylabel('Number of observations')
    plt.xlabel('Pareto set size (# of actions)')
    plt.title('Pareto-set size distribution')

    if do_save:
        plt.savefig(os.path.join(fig_path_, exp_name_ + '_counts.pdf'))

    plt.show()

    # Analyze accuracy of pareto set

    # Assign one if pareto set contains optimal action
    pareto_set_accuracy = [1 if y in x else 0 for x, y in zip(pareto_set, optimal_set)]
    pareto_mean_accuracy = statistics.mean(pareto_set_accuracy) * 100

    print(f'Overall {pareto_mean_accuracy} % of the pareto sets contain the action selected by the network trained' +
          f' using the correct reward function')

    if 'policy_actions' in actions_dict_:
        policy_accuracy = [1 if y == x else 0 for x, y in zip(optimal_set, policy_set)]
        policy_mean_accuracy = statistics.mean(policy_accuracy) * 100

        print(f'Overall {policy_mean_accuracy} % of the policy actions are the same as optimal actions')

    # Create results df to analyze accuracy by pareto-set size
    results_dict = {"Pareto Set Size": pareto_sizes, "Includes optimal": pareto_set_accuracy}
    results_df = pd.DataFrame(results_dict)

    results_df_grouped_mean = results_df.groupby('Pareto Set Size').mean()
    results_df_grouped_std = results_df.groupby('Pareto Set Size').std()

    print("Mean and std of pareto set accuracy by pareto-set size")

    print(results_df_grouped_mean.head(25))
    print(results_df_grouped_std.head(25))
