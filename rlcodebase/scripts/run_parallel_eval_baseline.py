import subprocess
import shlex

if __name__ == '__main__':
    n = 10
    seeds = range(1, n + 1)

    v = 6

    lr = 1e-4
    tuf = 8000

    cql_alphas = [0.001, 0.005, 0.01]

    r = 0

    logs = []
    for cql_alpha in cql_alphas:
        for seed in seeds:
            command = \
                'python rlcodebase/scripts/run_eval_baseline.py' + '\n' + \
                '--exp_name v%d_var1c_%d_offline_baseline_lr%.0e_tuf%g_cql%g_r%g_eval' \
                % (v, seed, lr, tuf, cql_alpha, r) + '\n' + \
                '--env_name MIMIC-Continuous' + '\n' + \
                '--env_rew_weights 1 0 0 0 0' + '\n' + \
                '--target_update_freq %g' % tuf + '\n' + \
                '--arch_dim 32' + '\n' + \
                '--baseline_file_prefix v%d_var1c_%d_offline_baseline_lr%.0e_tuf%g_cql%g_r%g_MIMIC' \
                % (v, seed, lr, tuf, cql_alpha, r) + '\n' + \
                '--offline --buffer_path "Replay_buffer_extraction/Encoded_paths13_all_rewards_var1.pkl"' + '\n' + \
                '--log_freq 1' + '\n' \
                '--seed %d' % seed

            print(command)

            args = shlex.split(command)
            log = subprocess.Popen(args)
            logs.append(log)

    for log in logs:
        log.wait()
