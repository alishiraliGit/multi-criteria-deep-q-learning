import subprocess
import shlex

if __name__ == '__main__':
    n = 10

    seeds = range(1, n + 1)

    lr = 1e-4
    tuf = 8000

    cql_alphas = [0.001, 0.005, 0.01]

    r = 0
    weights = [1.] + [r] * 4

    # cord = 3
    # weights = [0] * 5
    # weights[cord] = 1

    logs = []
    for cql_alpha in cql_alphas:
        for seed in seeds:
            command = \
                'python rlcodebase/scripts/run_dqn.py' + '\n' + \
                '--exp_name v6_var1c_%d_offline_baseline_lr%.0e_tuf%g_cql%g_r%g' % (seed, lr, tuf, cql_alpha, r) + '\n' + \
                '--env_name MIMIC-Continuous' + '\n' + \
                '--env_rew_weights %g %g %g %g %g' % tuple(weights) + '\n' + \
                '--offline --buffer_path "Replay_buffer_extraction/Encoded_paths13_all_rewards_var1.pkl"' + '\n' + \
                '--add_cql_loss --cql_alpha %g' % cql_alpha + '\n' + \
                '--double_q' + '\n' + \
                '--arch_dim 32' + '\n' + \
                '--target_update_freq %d' % tuf + '\n' \
                '--no_weights_in_path' + '\n' + \
                '--scalar_log_freq 500' + '\n' \
                '--params_log_freq 500' + '\n' \
                '--save_best' + '\n' + \
                '--seed %d' % seed

            print(command)

            args = shlex.split(command)
            log = subprocess.Popen(args)
            logs.append(log)

    for log in logs:
        log.wait()
