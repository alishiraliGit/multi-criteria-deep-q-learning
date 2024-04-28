import subprocess
import shlex

if __name__ == '__main__':
    """
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
    
    """
    # Discrete BCQ Baseline

    n = 5
    seeds = range(1, n + 1)

    v = 8
    lr = 1e-4
    tuf = 10

    bcq_thresholds = [0.01, 0.1, 0.3, 0.5]
    r = 0

    logs = []
    for bcq_thres in bcq_thresholds:
        for seed in seeds:
            command = \
                'python rlcodebase/scripts/run_eval_baseline.py' + '\n' + \
                '--exp_name v8_var1c_%d_offline_baseline_lr%.0e_tuf%g_bcq%g_r%g_polyak_eval' % (seed, lr, tuf, bcq_thres, r) + '\n' + \
                '--env_name MIMIC-Continuous' + '\n' + \
                '--env_rew_weights 1 0 0 0 0' + '\n' + \
                '--target_update_freq %g' % tuf + '\n' + \
                '--arch_dim 32' + '\n' + \
                '--baseline_file_prefix v8_var1c_%d_offline_baseline_lr%.0e_tuf%g_bcq%g_r%g_polyak' % (seed, lr, tuf, bcq_thres, r) + '\n' + \
                '--bcq --bcq_thres %g' % bcq_thres + '\n' + \
                '--polyak_target_update --tau 0.005' + '\n' + \
                '--offline --buffer_path "Replay_buffer_extraction/Encoded_paths13_all_rewards_var1.pkl"' + '\n' + \
                '--log_freq 1' + '\n' \
                '--seed %d' % seed

            print(command)

            args = shlex.split(command)
            log = subprocess.Popen(args)
            logs.append(log)

    for log in logs:
        log.wait()




    """


    ## Training

    #python rlcodebase/scripts/run_dqn.py --exp_name v7_b_var1c_1_offline_baseline_lr1.0e-4_tuf1_polyak_bcq0.3_r0 --env_name MIMIC-Continuous --env_rew_weights 1 0 0 0 0  
    --offline --buffer_path Replay_buffer_extraction/Encoded_paths13_all_rewards_var1.pkl --bcq --bcq_thres 0.3 
    --polyak_target_update --tau 0.005 --double_q --arch_dim 32 --target_update_freq 1 --no_weights_in_path 
    --scalar_log_freq 500 --params_log_freq 500 --save_best --seed 1
    
    #python rlcodebase/scripts/run_dqn.py --exp_name v7_b_var1c_1_offline_baseline_lr1.0e-4_tuf1_polyak_bcq0.1_r0 --env_name MIMIC-Continuous --env_rew_weights 1 0 0 0 0  --offline --buffer_path Replay_buffer_extraction/Encoded_paths13_all_rewards_var1.pkl --bcq --bcq_thres 0.1 --polyak_target_update --tau 0.005 --double_q --arch_dim 32 --target_update_freq 1 --no_weights_in_path --scalar_log_freq 500 --params_log_freq 500 --save_best --seed 1

            
    'python rlcodebase/scripts/run_dqn.py
    --exp_name v7_var1c_1_offline_baseline_lr1.0e-4_tuf1_polyak_bcq0.3_r0')
    --env_name MIMIC-Continuous
    --env_rew_weights 1 0 0 0 0 
    --offline --buffer_path "Replay_buffer_extraction/Encoded_paths13_all_rewards_var1.pkl"
    --bcq --bcq_thres 0.3 --polyak_target_update --tau 0.005
    --double_q
    --arch_dim 32
    --target_update_freq 1
    --no_weights_in_path
    --scalar_log_freq 500
    --params_log_freq 500
    --save_best
    --seed 1

    ## Evaluation

    #python rlcodebase/scripts/run_eval_baseline.py  --exp_name v7_var1c_1_offline_baseline_lr1.0e-4_tuf1_polyak_bcq0.3_r0_eval --env_name MIMIC-Continuous --env_rew_weights 1 0 0 0 0 --target_update_freq 1 --arch_dim 32 --baseline_file_prefix v7_var1c_1_offline_baseline_lr1.0e-4_tuf1_polyak_bcq0.3_r0_MIMIC --bcq --bcq_thres 0.3 --polyak_target_update --tau 0.005 --offline --buffer_path Replay_buffer_extraction/Encoded_paths13_all_rewards_var1.pkl --log_freq 1 --seed 1

    #v7_b_var1c_1_offline_baseline_lr1.0e-4_tuf1_polyak_bcq0.1_r0 

    #python rlcodebase/scripts/run_eval_baseline.py  --exp_name v7_var1c_1_offline_baseline_lr1.0e-4_tuf1_polyak_bcq0.3_r0_eval --env_name MIMIC-Continuous --env_rew_weights 1 0 0 0 0 --target_update_freq 1 --arch_dim 32 --baseline_file_prefix v7_var1c_1_offline_baseline_lr1.0e-4_tuf1_polyak_bcq0.3_r0_MIMIC --bcq --bcq_thres 0.3 --polyak_target_update --tau 0.005 --offline --buffer_path Replay_buffer_extraction/Encoded_paths13_all_rewards_var1.pkl --log_freq 1 --seed 1

    #python rlcodebase/scripts/run_eval_baseline.py  --exp_name v7_b_var1c_1_offline_baseline_lr1.0e-4_tuf1_polyak_bcq0.1_r0_eval --env_name MIMIC-Continuous --env_rew_weights 1 0 0 0 0 --target_update_freq 1 --arch_dim 32 --baseline_file_prefix v7_b_var1c_1_offline_baseline_lr1.0e-4_tuf1_polyak_bcq0.1_r0_MIMIC --bcq --bcq_thres 0.1 --polyak_target_update --tau 0.005 --offline --buffer_path Replay_buffer_extraction/Encoded_paths13_all_rewards_var1.pkl --log_freq 1 --seed 1

    'python rlcodebase/scripts/run_eval_baseline.py 
    --exp_name v7_var1c_1_offline_baseline_lr1.0e-4_tuf1_polyak_bcq0.3_r0_eval
    --env_name MIMIC-Continuous
    --env_rew_weights 1 0 0 0 0
    --target_update_freq 1
    --arch_dim 32
    --bcq --bcq_thres 0.3 --polyak_target_update --tau 0.005
    --baseline_file_prefix v7_var1c_1_offline_baseline_lr1.0e-4_tuf1_polyak_bcq0.3_r0_MIMIC' 
    --offline --buffer_path "Replay_buffer_extraction/Encoded_paths13_all_rewards_var1.pkl"
    --log_freq 1
    --seed 1
    
    """
    
    
    
