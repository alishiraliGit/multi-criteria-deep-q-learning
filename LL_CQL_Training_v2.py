"""
#Phase 1 - CQL - Baseline

python cs285/scripts/run_dqn.py \
--exp_name p5_baseline \
--env_name LunarLander-Customizable \
--env_rew_weights 0 0 0 0 1 \
--cql \
--double_q --seed 1

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
--prune_with_icql \ #still need to build this functionality
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

seeds = list(range(31))
seeds = seeds[16:]

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
"""

command_stem = [
"python cs285/scripts/run_dqn.py --exp_name p51_pruned_icql_sparse_{eps_print} --env_name LunarLander-Customizable --env_rew_weights 0 0 0 0 1 --prune_with_icql --cql --pruning_file_prefix p5_LunarLander-Customizable --pruning_eps {eps} --double_q --seed 1 --no_weights_in_path" 
]

eps_list = [0.3,0.5]

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
