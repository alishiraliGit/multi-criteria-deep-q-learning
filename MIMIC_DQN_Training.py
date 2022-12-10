import shlex, subprocess

command_stem = [
"python cs285/scripts/run_dqn.py --exp_name offline_{r} --env_name LunarLander-Customizable --env_rew_weights 1 1 1 1 1 --double_q --seed 1 --offline_RL True --buffer_path {p}"
]


name = ['SOFA1','SOFA1c','SOFA2','SOFA2c','SOFAd2','lac1','lac1c','lac2','lac2c']#'baseline',
reward_path = ['./Replay_buffer_extraction/Paths_Reward_SOFA_1_binary.pkl','./Replay_buffer_extraction/Paths_Reward_SOFA_1_continous.pkl', 
    './Replay_buffer_extraction/Paths_Reward_SOFA_2_binary.pkl','./Replay_buffer_extraction/Paths_Reward_SOFA_2_continous.pkl','./Replay_buffer_extraction/Paths_Reward_SOFA_chnage2_binary.pkl',
    './Replay_buffer_extraction/Paths_Reward_lac_1_binary.pkl','./Replay_buffer_extraction/Paths_Reward_lac_1_continous.pkl',
    './Replay_buffer_extraction/Paths_Reward_lac_2_binary.pkl','./Replay_buffer_extraction/Paths_Reward_lac_2_continous.pkl']#'./Replay_buffer_extraction/Paths_sparse_90d_rew.pkl',

commands = []
for command in command_stem:
    for i in range(len(name)):
        commands.append(command.format(r=name[i],p=reward_path[i]))

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