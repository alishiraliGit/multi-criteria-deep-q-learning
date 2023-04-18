import shlex, subprocess

if __name__ == '__main__':
    n = 3 #5 used for baseline
    seeds = range(1, n + 1)

    #Unused hyperparameters

    # lr1 = 1e-5
    # tuf1 = 1000
    # lr2 = 1e-4
    # tuf2 = 8000
    # cql_alpha = 0.001

    r = 10
    w_bound = [r]*3
    
    consistency_alphas = [20, 40, 160] #80,
    noise_level = [0,0.1,1,10]
    logs = []
    
    ####################################
    # #baseline loop
    ####################################

    # for seed in seeds:
    #     command = \
    #         'python rlcodebase/scripts/run_dqn.py' + '\n' + \
    #         '--exp_name v7LL_%d_baseline_sparse ' \
    #         % (seed) + '\n' + \
    #         '--env_name LunarLander-Customizable' + '\n' + \
    #         '--env_rew_weights 0 0 0 0 1' + '\n' + \
    #         '--double_q' + '\n' + \
    #         '--no_weights_in_path' + '\n' + \
    #         '--save_best' + '\n' + \
    #         '--seed %d' % seed

    #     print(command)

    #     args = shlex.split(command)
    #     log = subprocess.Popen(args)
    #     log.wait()
    #     #log = ParallelExecutor(max_procs=2).execute(args)
    #     logs.append(log)

    ####################################
    # Pruned DQN Phase 1 Loop
    ####################################

    for consistency_alpha in consistency_alphas:
        for noise in noise_level:
            for seed in seeds:
                command = \
                    'python rlcodebase/scripts/run_dqn.py' + '\n' + \
                    '--exp_name v7LL_%d_cmdqn_noise%g_r%g' \
                    % (seed, noise, r) + '\n' + \
                    '--env_name LunarLander-MultiInterRewardNoise' + '\n' + \
                    '--consistent_mdqn' + '\n' + \
                    '--w_bound %g %g %g' % tuple(w_bound) + '\n' + \
                    '--consistency_alpha %g' % consistency_alpha  + '\n' + \
                    '--env_noise_level %g' % noise + '\n' + \
                    '--double_q' + '\n' + \
                    '--no_weights_in_path' + '\n' +  \
                    '--save_best' + '\n' + \
                    '--seed %d' % seed

                print(command)

                args = shlex.split(command)
                log = subprocess.Popen(args)
                log.wait()
                logs.append(log)
    
    ####################################
    # Pruned DQN Phase 2 Loop
    ####################################

    for consistency_alpha in consistency_alphas:
        for noise in noise_level:
            for seed in seeds:
                command = \
                    'python rlcodebase/scripts/run_dqn.py' + '\n' + \
                    '--exp_name v7LL_%d_pruned_cmdqn_noise%g_r%g_sparse' \
                    % (seed, noise, r) + '\n' + \
                    '--env_name LunarLander-Customizable' + '\n' + \
                    '--env_rew_weights 0 0 0 0 1' + '\n' + \
                    '--prune_with_mdqn' + '\n' + \
                    '--v7LL_%d_cmdqn_noise%g_r%g' \
                    % (seed, noise, r) + '\n' + \
                    '--pruning_n_draw 20' + '\n' + \
                    '--double_q' + '\n' + \
                    '--no_weights_in_path' + '\n' +  \
                    '--save_best' + '\n' + \
                    '--seed %d' % seed

                print(command)

                args = shlex.split(command)
                log = subprocess.Popen(args)
                log.wait()
                logs.append(log)
    
    for log in logs:
        log.wait()

# List of commands for LunarLander Experiments

# Baseline:
# python rlcodebase/scripts/run_dqn.py \
# --exp_name v6_[SEED]_baseline_sparse \
# --env_name LunarLander-Customizable \
# --env_rew_weights 0 0 0 0 1 \
# --double_q \
# --no_weights_in_path \
# --save_best \
# --seed [SEED]


# Phase 1:
# python rlcodebase/scripts/run_dqn.py \
# --exp_name v7LL_[SEED]_cmdqn \
# --env_name LunarLander-MultiInterRewardNoise \
# --consistent_mdqn \
# --w_bound 10 \
# --consistency_alpha [20, 40, 80, 160] \
# --double_q \
# --no_weights_in_path \
# --save_best \
# --seed [SEED]

# Phase 2:
# python rlcodebase/scripts/run_dqn.py \
# --exp_name v6_[SEED]_offline_pruned_cmdqn_sparse \
# --env_name LunarLander-Customizable \
# --env_rew_weights 0 0 0 0 1 \
# --prune_with_mdqn \
# --pruning_file_prefix v6_[SEED]_cmdqn_LunarLander \
# --pruning_n_draw 20 \
# --double_q \
# --no_weights_in_path \
# --save_best \
# --seed [SEED]

"""
eps [0.01 0.03 0.05 0.0 0.15 0.1 0.2 0.3]
Pareto set accuacies for some models [96.95517525, 96.95517525, 91.47543778, 96.95517525, 78.67107369, 82.18469727, 72.467776, 62.38149807]
Mean Pareto set size [5.1825569, 5.1825569, 4.4784684, 5.1825569, 3.11385656, 3.47977536, 2.56945326, 1.90034 ]
Action space dim is 6
Chance level [0.86375948, 0.86375948, 0.7464114 , 0.86375948, 0.51897609, 0.57996256, 0.42824221, 0.31672333]

#Phase 1 - DQN - baseline

python rlcodebase/scripts/run_dqn.py \
--exp_name p4_baseline \
--env_name LunarLander-Customizable \
--env_rew_weights 0 0 0 0 1 \
--double_q --seed 1

#Phase 1 - DQN

python rlcodebase/scripts/run_dqn.py \
--exp_name p4 \
--env_name LunarLander-Customizable \
--env_rew_weights 1 1 1 1 0 \
--double_q --seed 1

#Phase 2 - DQN

python rlcodebase/scripts/run_dqn.py \
--exp_name p4_pruned_idqn_sparse \
--env_name LunarLander-Customizable \
--env_rew_weights 0 0 0 0 1 \
--prune_with_idqn \
--pruning_file_prefix p4_LunarLander-Customizable \
--pruning_eps 0.3 \
--double_q --seed 1 \
--no_weights_in_path
"""

# List of commands for LunarLander Experiments

# Baseline:
# python rlcodebase/scripts/run_dqn.py \
# --exp_name v6_[SEED]_baseline_sparse \
# --env_name LunarLander-Customizable \
# --env_rew_weights 0 0 0 0 1 \
# --double_q \
# --no_weights_in_path \
# --save_best \
# --seed [SEED]


# Phase 1:
# python rlcodebase/scripts/run_dqn.py \
# --exp_name v6_[SEED]_cmdqn \
# --env_name LunarLander-MultiInterReward \
# --consistent_mdqn \
# --w_bound [5, 10] \
# --consistency_alpha [20, 40, 80, 160] \
# --double_q \
# --save_best \
# --seed [SEED]

# Phase 2:
# python rlcodebase/scripts/run_dqn.py \
# --exp_name v6_[SEED]_offline_pruned_cmdqn_sparse \
# --env_name LunarLander-Customizable \
# --env_rew_weights 0 0 0 0 1 \
# --prune_with_mdqn \
# --pruning_file_prefix v6_[SEED]_cmdqn_LunarLander \
# --pruning_n_draw 20 \
# --double_q \
# --no_weights_in_path \
# --save_best \
# --seed [SEED]

# python rlcodebase/scripts/run_dqn.py \
# --exp_name v6_[SEED]_baseline_sparse \
# --env_name LunarLander-Customizable \
# --env_rew_weights 0 0 0 0 1 \
# --double_q \
# --no_weights_in_path \
# --save_best \
# --seed [SEED]

# command_stem = [
# "python rlcodebase/scripts/run_dqn.py --exp_name p4_pruned_idqn_sparse_{eps_print} --env_name LunarLander-Customizable --env_rew_weights 0 0 0 0 1 --prune_with_icql --pruning_file_prefix p4_LunarLander-Customizable --pruning_eps {eps} --double_q --seed 1 --no_weights_in_path"
# ]

# eps_data = [0, 0.1, 0.2, 0.5]

# commands = []
# for command in command_stem:
#     for i in range(len(eps_data)):
#         commands.append(command.format(eps=eps_data[i], eps_print = eps_data[i]*100))

# if __name__ == "__main__":
#     for command in commands:
#         print(command)
#     user_input = None
#     while user_input not in ['y', 'n']:
#         user_input = input('Run experiment with above commands? (y/n): ')
#         user_input = user_input.lower()[:1]
#     if user_input == 'n':
#         exit(0)
#     for command in commands:
#         args = shlex.split(command)
#         process = subprocess.Popen(args)
#         process.wait()

#python rlcodebase/scripts/post_process_training_logs.py --prefix p4_pruned --y_tag Train_AverageReturn --x_tag Train_EnvstepsSoFar --baseline_model p4_baseline