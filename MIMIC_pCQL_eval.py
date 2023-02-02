import shlex, subprocess

command_stem = [
#"python cs285/scripts/run_eval_pareto_opt_dqn.py --exp_name pCQLvdl{eps}_eval --env_name MIMIC --pruning_file_prefix MIMICCQL_ --pruning_eps {e} --env_rew_weights 1 0 0 0 0 0 0 0 0 0 0 --seed 1 --offline --cql --no_weights_in_path --buffer_path './Replay_buffer_extraction/Encoded_paths3_all_rewards.pkl'",
"python cs285/scripts/run_eval_pruning.py --exp_name pCQLv2{eps}_eval --env_name MIMIC --pruning_file_prefix MIMICCQL_ --pruning_eps {e} --env_rew_weights 1 0 0 0 0 0 0 0 0 0 0 --seed 1 --offline --cql --no_weights_in_path --buffer_path './Replay_buffer_extraction/Encoded_paths3_all_rewards.pkl' --trained_pruning_critic pCQLv2_{eps}"
]

eps_list = [0,0.05,0.1,0.2,0.3, 0.5]

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

#python cs285/scripts/run_eval_pruning.py --exp_name test_eval --env_name MIMIC --pruning_file_prefix MIMICCQL_ --pruning_eps 0.1 --env_rew_weights 1 0 0 0 0 0 0 0 0 0 0 --seed 1 --offline --cql --no_weights_in_path --buffer_path './Replay_buffer_extraction/Encoded_paths3_all_rewards.pkl'
#python cs285/scripts/post_process_eval_pruning.py --prefix pCQLv4*_eval --pruning_file_prefix MIMICCQLv4_ --show --critic_prefix pCQLv4_ --pruned --prune_with_icql --cql --seed 1 --env_rew_weights 1 0 0 0 0 0 0 0 0 0 0 --buffer_path './Replay_buffer_extraction/Encoded_paths13_all_rewards.pkl'

#python cs285/scripts/run_eval_pruning.py --exp_name test_eval --env_name MIMIC --phase_2_critic_file_prefix pCQLv4_10 --pruning_file_prefix MIMICCQLv4_ --pruning_eps 0.1 --prune_with_icql --env_rew_weights 1 0 0 0 0 0 0 0 0 0 0 --seed 1 --offline --no_weights_in_path --buffer_path './Replay_buffer_extraction/Encoded_paths13_all_rewards.pkl'