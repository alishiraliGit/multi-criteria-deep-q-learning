import shlex, subprocess

command_stem = [
"python cs285/scripts/run_dqn.py --exp_name MIMIC_{r} --env_name MIMIC --env_rew_weights {w} --double_q --seed 1 --offline --no_weights_in_path --buffer_path './Replay_buffer_extraction/Encoded_paths_all_rewards.pkl'"
]



name = ['baseline','SOFA1','SOFA1c','SOFA2','SOFA2c','SOFAd2','lac1','lac1c','lac2','lac2c']#

reward_weight = ['1 0 0 0 0 0 0 0 0 0 0', '0 0 1 0 0 0 0 0 0 0 0', '0 0 0 1 0 0 0 0 0 0 0', '0 0 0 0 1 0 0 0 0 0 0', '0 0 0 0 0 0 1 0 0 0 0', 
               '0 0 0 0 0 0 1 0 0 0 0', '0 0 0 0 0 0 0 1 0 0 0', '0 0 0 0 0 0 0 0 1 0 0', '0 0 0 0 0 0 0 0 0 1 0', '0 0 0 0 0 0 0 0 0 0 1']#'./Replay_buffer_extraction/Paths_sparse_90d_rew.pkl',

"""
reward_path = ['./Replay_buffer_extraction/Paths_Reward_SOFA_1_binary.pkl','./Replay_buffer_extraction/Paths_Reward_SOFA_1_continous.pkl', 
    './Replay_buffer_extraction/Paths_Reward_SOFA_2_binary.pkl','./Replay_buffer_extraction/Paths_Reward_SOFA_2_continous.pkl','./Replay_buffer_extraction/Paths_Reward_SOFA_chnage2_binary.pkl',
    './Replay_buffer_extraction/Paths_Reward_lac_1_binary.pkl','./Replay_buffer_extraction/Paths_Reward_lac_1_continous.pkl',
    './Replay_buffer_extraction/Paths_Reward_lac_2_binary.pkl','./Replay_buffer_extraction/Paths_Reward_lac_2_continous.pkl']#'./Replay_buffer_extraction/Paths_sparse_90d_rew.pkl',
"""

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