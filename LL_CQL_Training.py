"""
eps [0.0, 0.1 0.2 0.3 0.05]
Pareto set accuacies for some models [93.5283072  74.23434419 62.85043658 55.17603981 86.47357056]
Mean Pareto set size [5.10381185, 3.28187964, 2.22081495, 1.67816167, 4.36744907] 
Action space dim is 6
Chance level [0.85063531, 0.54697994, 0.37013582, 0.27969361, 0.72790818]

#Phase 1 - CQL

python cs285/scripts/run_dqn.py \
--exp_name p5 \
--env_name LunarLander-Customizable \
--env_rew_weights 1 1 1 1 0 \
--cql \
--double_q --seed 1

#Phase 2 - DQN

python cs285/scripts/run_dqn.py \
--exp_name p4_pruned_idqn_sparse \
--env_name LunarLander-Customizable \
--env_rew_weights 0 0 0 0 1 \
--prune_with_idqn \
--pruning_file_prefix p4_LunarLander-Customizable \
--pruning_eps 0.3 \
--double_q --seed 1 \
--no_weights_in_path

#Phase 2 - CQL

python cs285/scripts/run_dqn.py \
--exp_name p5_pruned_icql_sparse \
--env_name LunarLander-Customizable \
--env_rew_weights 0 0 0 0 1 \
--prune_with_icql \
--cql \
--pruning_file_prefix p5_LunarLander-Customizable \
--pruning_eps 0.3 \
--double_q --seed 1 \
--no_weights_in_path
"""


import shlex, subprocess

"""
command_stem = [
"python cs285/scripts/run_dqn.py --exp_name p5 --env_name LunarLander-Customizable --env_rew_weights 1 1 1 1 0 --cql --double_q --seed {s}" 
]

seeds = list(range(16))
seeds = seeds[1:]

commands = []
for command in command_stem:
    for i in range(len(seeds)):
        commands.append(command.format(s=seeds[i]))

if __name__ == "__main__":
    for command in commands:
        print(command)
    user_input = None
    while user_input not in ['y', 'n']:
        user_input = input('Run experiment with above commands? (y/n): ')
        user_input = user_input.lower()[:1]
    if user_input == 'n':
        exit(0)
    for command in commands:
        args = shlex.split(command)
        process = subprocess.Popen(args)
        process.wait()


command_stem = [
"python cs285/scripts/run_dqn.py --exp_name p51_pruned_icql_sparse_{eps_print} --env_name LunarLander-Customizable --env_rew_weights 0 0 0 0 1 --prune_with_icql --cql --pruning_file_prefix p5_LunarLander-Customizable --pruning_eps {eps} --double_q --seed 1 --no_weights_in_path" 
]

eps_list = [0.05,0.1,0.2]

commands = []
for command in command_stem:
    for i in range(len(eps_list)):
        commands.append(command.format(eps_print=eps_list[i]*100, eps=eps_list[i]))

if __name__ == "__main__":
    for command in commands:
        print(command)
    user_input = None
    while user_input not in ['y', 'n']:
        user_input = input('Run experiment with above commands? (y/n): ')
        user_input = user_input.lower()[:1]
    if user_input == 'n':
        exit(0)
    for command in commands:
        args = shlex.split(command)
        process = subprocess.Popen(args)
        process.wait()


"""
command_stem = [
"python cs285/scripts/run_eval_pruning.py --exp_name p5_icql_eval_{eps_print} --env_name LunarLander-Customizable --env_rew_weights 0 0 0 0 1 --opt_file_prefix p4_opt --prune_with_icql --pruning_file_prefix p5_L --pruning_eps {eps} --seed 1" 
]

eps_list = [0, 0.05,0.1,0.2, 0.3]

commands = []
for command in command_stem:
    for i in range(len(eps_list)):
        commands.append(command.format(eps_print=eps_list[i]*100, eps=eps_list[i]))

if __name__ == "__main__":
    for command in commands:
        print(command)
    user_input = None
    while user_input not in ['y', 'n']:
        user_input = input('Run experiment with above commands? (y/n): ')
        user_input = user_input.lower()[:1]
    if user_input == 'n':
        exit(0)
    for command in commands:
        args = shlex.split(command)
        process = subprocess.Popen(args)
        process.wait()

#python cs285/scripts/post_process_training_logs.py --prefix p5_pruned --y_tag Train_AverageReturn --x_tag Train_EnvstepsSoFar --baseline_model p5_baseline

#python cs285/scripts/post_process_training_logs.py --prefix p5_pruned_icql_sparse_30 --y_tag Train_AverageReturn --x_tag Train_EnvstepsSoFar --baseline_model p5_baseline

#python cs285/scripts/run_eval_pruning.py --exp_name p5_icql_eval --env_name LunarLander-Customizable --env_rew_weights 0 0 0 0 1 --opt_file_prefix p4_opt --prune_with_icql #TODO --pruning_file_prefix p5_pruned --pruning_eps w --seed 1