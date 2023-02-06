import subprocess
import shlex


if __name__ == '__main__':
    n = 3

    seed = range(1, n + 1)
    consistency_alpha = 20
    cql_alpha = 0.001
    r = 1

    logs = []
    for idx in range(n):
        command = \
            'python run_dqn.py' + '\n' \
            '--exp_name expvar1cfast_%d_offline_pruned_cmdqn_alpha%g_cql%g_r%g_sparse' % (seed[idx], consistency_alpha, cql_alpha, r) + '\n' \
            '--env_name MIMIC' + '\n' \
            '--env_rew_weights 1 0 0 0 0 0' + '\n' \
            '--offline --buffer_path "../../Replay_buffer_extraction/Encoded_paths13_all_rewards_var1.pkl"' + '\n' \
            '--prune_with_mdqn' + '\n' \
            '--pruning_file_prefix expvar1c_%d_offline_cmdqn_alpha%g_cql%g_r%g_MIMIC' % (seed[idx], consistency_alpha, cql_alpha, r) + '\n' \
            '--add_cql_loss --cql_alpha %g' % cql_alpha + '\n' \
            '--double_q' + '\n' \
            '--arch_dim 32' + '\n' \
            '--no_weights_in_path' + '\n' \
            '--scalar_log_freq 500' + '\n' \
            '--params_log_freq 500' + '\n' \
            '--save_best' + '\n' \
            '--seed %d' % seed[idx]

        print(command)

        args = shlex.split(command)
        log = subprocess.Popen(args)
        logs.append(log)

    for log in logs:
        log.wait()
