import subprocess
import sys
import os
import shlex

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

if __name__ == '__main__':
    n = 3

    seed = range(1, n + 1)
    r = 0.001
    cql_alpha = 0.001

    # weights = np.array([r] + [1]*5)/np.sqrt(r**2 + 5)
    weights = np.array([1.] + [r] * 5)
    # cord = 5
    # weights = np.zeros((6,))
    # weights[cord] = 1

    tuf = 8000

    logs = []
    for idx in range(n):
        command = \
            'python run_dqn.py' + '\n' + \
            '--exp_name expvar1clr-4_%d_offline_baseline_cql%g_r%g_tuf%d' % (seed[idx], cql_alpha, r, tuf) + '\n' + \
            '--env_name MIMIC' + '\n' + \
            '--env_rew_weights %g %g %g %g %g %g' % tuple(weights.tolist()) + '\n' + \
            '--offline --buffer_path "../../Replay_buffer_extraction/Encoded_paths13_all_rewards_var1.pkl"' + '\n' + \
            '--add_cql_loss --cql_alpha %g' % cql_alpha + '\n' + \
            '--double_q' + '\n' + \
            '--arch_dim 32' + '\n' + \
            '--no_weights_in_path' + '\n' + \
            '--scalar_log_freq 500' + '\n' \
            '--params_log_freq 500' + '\n' \
            '--save_best' + '\n' + \
            '--seed %d' % seed[idx]

        print(command)

        args = shlex.split(command)
        log = subprocess.Popen(args)
        logs.append(log)

    for log in logs:
        log.wait()
