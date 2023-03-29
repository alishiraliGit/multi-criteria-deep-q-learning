import os
import pickle
import glob
import statistics

if __name__ == "__main__":
    do_save = False

    # Path settings
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data')
    fig_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'figs')
    if not (os.path.exists(fig_path)):
        os.makedirs(fig_path)

    exp_name = 'againexpvar1clr-4_3_offline_baseline_cql0.01_r0_tuf8000_beval'

    all_folders = glob.glob(os.path.join(data_path, exp_name + '*'))
    if len(all_folders) > 1:
        raise Exception('More than one folder with this exp_name prefix found!')
    if len(all_folders) == 0:
        raise Exception('No such file found!')

    folder_path = all_folders[0]
    file_path = os.path.join(folder_path, 'actions.pkl')

    # Load actions
    with open(file_path, 'rb') as f:
        actions_dict = pickle.load(f)

    opt_actions = actions_dict['opt_actions']
    policy_actions = actions_dict['policy_actions']

    # Merge all trajectories
    optimal_set = []
    for opt_actions_per_path in opt_actions:
        optimal_set += opt_actions_per_path

    policy_set = []
    for pol_actions_per_path in policy_actions:
        policy_set += pol_actions_per_path

    # Analyze accuracy
    policy_accuracy = [1 if y == x else 0 for x, y in zip(optimal_set, policy_set)]
    policy_mean_accuracy = statistics.mean(policy_accuracy) * 100

    print(f'Overall {policy_mean_accuracy} % of the policy actions are the same as optimal actions')

