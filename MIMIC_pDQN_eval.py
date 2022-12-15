import shlex, subprocess


command_stem = [
"python cs285/scripts/run_eval_pareto_opt_dqn.py --exp_name pDQNvdl{eps}_eval --env_name MIMIC --pruning_file_prefix MIMICvdl_ --pruning_eps {e} --env_rew_weights 1 0 0 0 0 0 0 0 0 0 0 --seed 1 --offline --no_weights_in_path --buffer_path './Replay_buffer_extraction/Encoded_paths3_all_rewards.pkl'"
]

eps_list = [0,0.05,0.1,0.2,0.3]

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

