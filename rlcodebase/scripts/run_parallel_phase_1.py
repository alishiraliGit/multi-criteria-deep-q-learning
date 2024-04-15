import subprocess
import shlex


if __name__ == '__main__':
    n = 10
    seeds = range(1, n + 1)

    lr = 1e-5
    tuf = 1000

    cql_alpha = 0.001

    consistency_alphas = [20, 40, 80, 160]

    r = 10
    w_bound = [r]*5

    logs = []
    for consistency_alpha in consistency_alphas:
        for seed in seeds:
            command = \
                'python rlcodebase/scripts/run_dqn.py' + '\n' + \
                '--exp_name v6_var1c_%d_offline_cmdqn_lr%.0e_tuf%g_cql%g_alpha%g_r%g' \
                % (seed, lr, tuf, cql_alpha, consistency_alpha, r) + '\n' + \
                '--env_name MIMIC-MultiContinuousReward' + '\n' + \
                '--offline --buffer_path "Replay_buffer_extraction/Encoded_paths13_all_rewards_var1.pkl"' + '\n' + \
                '--consistent_mdqn' + '\n' + \
                '--w_bound %g %g %g %g %g' % tuple(w_bound) + '\n' + \
                '--consistency_alpha %g' % consistency_alpha + '\n' + \
                '--add_cql_loss --cql_alpha %g' % cql_alpha + '\n' + \
                '--double_q' + '\n' + \
                '--arch_dim 32' + '\n' + \
                '--target_update_freq %d' % tuf + '\n' + \
                '--scalar_log_freq 200' + '\n' + \
                '--params_log_freq 200' + '\n' + \
                '--save_best' + '\n' + \
                '--no_gpu' + '\n' + \
                '--seed %d' % seed

            print(command)

            args = shlex.split(command)
            log = subprocess.Popen(args)
            # Uncomment in case parallel processing is not possible on local machine
            # log.wait()
            logs.append(log)

    for log in logs:
        log.wait()
