import subprocess
import shlex

import numpy as np


if __name__ == '__main__':
    n = 3
    seed = range(1, n + 1)

    consistency_alpha = 20
    r = 10
    w_bound = np.array([1.] + [r] * 5)

    cql_alpha = 0.001

    tuf = 8000

    logs = []
    for idx in range(n):
        command = \
            'python run_dqn.py' + '\n' \
            '--exp_name expvar1clr-5_%d_offline_cmdqn_alpha%g_cql%g_r%g_tuf%g' % (seed[idx], consistency_alpha, cql_alpha, r, tuf) + '\n' \
            '--env_name MIMIC-MultiContinuousReward' + '\n' \
            '--offline --buffer_path "../../Replay_buffer_extraction/Encoded_paths13_all_rewards_var1.pkl"' + '\n' \
            '--consistent_mdqn' + '\n' \
            '--w_bound %g %g %g %g %g %g' % tuple(w_bound.tolist()) + '\n' \
            '--consistency_alpha %g' % consistency_alpha + '\n' \
            '--add_cql_loss --cql_alpha %g' % cql_alpha + '\n' \
            '--double_q' + '\n' \
            '--arch_dim 32' + '\n' \
            '--target_update_freq %d' % tuf + '\n' \
            '--scalar_log_freq 200' + '\n' \
            '--params_log_freq 200' + '\n' \
            '--save_best' + '\n' \
            '--no_gpu' + '\n' \
            '--seed %d' % seed[idx] #added no_gpu to allow more parallel runs on local machine

        print(command)

        args = shlex.split(command)
        log = subprocess.Popen(args)
        log.wait() #added in case parallel processing is not possible on local machine
        logs.append(log)
        

    for log in logs:
        log.wait()
