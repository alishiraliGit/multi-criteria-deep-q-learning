import shlex, subprocess

"""
command_stem = [
"python rlcodebase/scripts/run_dqn.py --exp_name MIMICCQL_{r} --env_name MIMIC --env_rew_weights {w} --double_q --seed 1 --scalar_log_freq 2000 --params_log_freq 2000 --offline --cql --no_weights_in_path --buffer_path './Replay_buffer_extraction/Encoded_paths3_all_rewards.pkl'"
]


#just reweighting of rewards
name = ['SL1bw12','SL1bw21','SL1cw12','SL1cw21','Sd2L1bw12',
        'Sd2L1bw21','Sd2Lbcw12','Sd2Lbcw21','allSLw12','allSLw21']#

reward_weight = ['0 0 1 0 0 0 0 2 0 0 0', '0 0 2 0 0 0 0 1 0 0 0', '0 0 0 1 0 0 0 0 2 0 0', '0 0 0 2 0 0 0 0 1 0 0', '0 0 0 0 0 0 1 2 0 0 0', 
               '0 0 0 0 0 0 2 1 0 0 0', '0 0 0 0 0 0 1 2 2 0 0', '0 0 0 0 0 0 2 1 1 0 0', '0 0 1 1 0 0 1 2 2 0 0', '0 0 2 2 0 0 2 1 1 0 0']


#name = ['baseline','SL1','SOFA1c','SOFA2','SOFA2c',
#       'SOFAd2','lac1','lac1c','lac2','lac2c']#

#reward_weight = ['1 0 0 0 0 0 0 0 0 0 0', '0 0 1 0 0 0 0 0 0 0 0', '0 0 0 1 0 0 0 0 0 0 0', '0 0 0 0 1 0 0 0 0 0 0', '0 0 0 0 0 1 0 0 0 0 0', 
               #'0 0 0 0 0 0 1 0 0 0 0', '0 0 0 0 0 0 0 1 0 0 0', '0 0 0 0 0 0 0 0 1 0 0', '0 0 0 0 0 0 0 0 0 1 0', '0 0 0 0 0 0 0 0 0 0 1']

#path['reward'] = weights[0]*path['sparse_90d_rew'] + weights[1]*path['Reward_matrix_paper'] + weights[2]*path['Reward_SOFA_1_continous'] + weights[3]*path['Reward_SOFA_1_binary'] + \
#                        weights[4]*path['Reward_SOFA_2_continous'] + weights[5]*path['Reward_SOFA_2_binary'] + weights[6]*path['Reward_SOFA_change2_binary'] + weights[7]*path['Reward_lac_1_continous'] + \
#                        weights[8]*path['Reward_lac_1_binary'] + weights[9]*path['Reward_lac_2_continous'] + weights[10]*path['Reward_lac_2_binary'] 


commands = []
for command in command_stem:
    for i in range(len(name)):
        commands.append(command.format(r=name[i],w=reward_weight[i]))

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
"python rlcodebase/scripts/run_dqn.py --exp_name pCQLv2_{eps} --env_name MIMIC --pruning_file_prefix MIMICCQL_*_Top --pruning_eps {e} --env_rew_weights 1 0 0 0 0 0 0 0 0 0 0 --double_q --seed 1 --scalar_log_freq 2000 --params_log_freq 2000 --offline --cql --no_weights_in_path --buffer_path './Replay_buffer_extraction/Encoded_paths3_all_rewards.pkl'"
]

eps_list = [0.2, 0.3, 0.5]

commands = []
for command in command_stem:
    for i in range(len(eps_list)):
        commands.append(command.format(eps=int(eps_list[i]*100), e=eps_list[i]))

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
"python rlcodebase/scripts/run_dqn.py --exp_name pCQLvdl_{eps}ep --env_name MIMIC --pruning_file_prefix MIMICCQL_ --pruning_eps {e} --env_rew_weights 1 0 0 0 0 0 0 0 0 0 0 --double_q --seed 1 --scalar_log_freq 2000 --params_log_freq 2000 --offline --cql --no_weights_in_path --buffer_path './Replay_buffer_extraction/Encoded_paths3_all_rewards.pkl'"
]

alpha_list = [0.1,0.3]

commands = []
for command in command_stem:
    for i in range(len(alpha_list)):
        commands.append(command.format(eps=int(alpha_list[i]*100), e=alpha_list[i]))

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