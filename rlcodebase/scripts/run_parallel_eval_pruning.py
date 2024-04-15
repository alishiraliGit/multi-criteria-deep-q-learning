import subprocess
import shlex


if __name__ == '__main__':
    n = 10
    seeds = range(1, n + 1)

    lr1 = 1e-5
    tuf1 = 1000
    r1 = 10

    lr2 = 1e-4
    tuf2 = 8000

    cql_alpha = 0.001
    consistency_alphas = [20, 40, 80, 160]

    logs = []
    for consistency_alpha in consistency_alphas:
        for seed in seeds:
            command = \
                'python rlcodebase/scripts/run_eval_pruning.py' + '\n' + \
                '--exp_name v6_var1c_%d_offline_pruned_cmdqn_lr[1]%.0e_tuf[1]%g_r[1]%g_lr[2]%.0e_tuf[2]%g_cql%g_alpha%g_sparse_eval' \
                % (seed, lr1, tuf1, r1, lr2, tuf2, cql_alpha, consistency_alpha) + '\n' + \
                '--env_name MIMIC-Continuous' + '\n' + \
                '--env_rew_weights 1 0 0 0 0' + '\n' + \
                '--target_update_freq %g' % tuf2 + '\n' + \
                '--arch_dim 32' + '\n' + \
                '--phase_2_critic_file_prefix v6_var1c_%d_offline_pruned_cmdqn_lr[1]%.0e_tuf[1]%g_r[1]%g_lr[2]%.0e_tuf[2]%g_cql%g_alpha%g_sparse_MIMIC' \
                % (seed, lr1, tuf1, r1, lr2, tuf2, cql_alpha, consistency_alpha) + '\n' + \
                '--pruning_file_prefix v6_var1c_%d_offline_cmdqn_lr%.0e_tuf%g_cql%g_alpha%g_r%g_MIMIC' \
                % (seed, lr1, tuf1, cql_alpha, consistency_alpha, r1) + '\n' + \
                '--prune_with_mdqn' + '\n' + \
                '--pruning_n_draw 100' + '\n' + \
                '--offline --buffer_path "Replay_buffer_extraction/Encoded_paths13_all_rewards_var1.pkl"' + '\n' + \
                '--log_freq 1' + '\n' + \
                '--seed %d' % seed

            print(command)

            args = shlex.split(command)
            log = subprocess.Popen(args)
            logs.append(log)

    for log in logs:
        log.wait()
