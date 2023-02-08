## Setup
I recommend creating a new virtual environment for this project and
install the requirements directly from [requirements.txt](requirements.txt).

## Environment specification
Currently, only LunarLander for online setting and MIMIC-III for offline setting are available. 

### Customized rewards
You can run DQN on LunarLander with customized rewards by choosing `LunarLander-Customizable`
for the `env_name` and specifying reward weights in the `env_rew_weights` argument:
```shell
python cs285/scripts/run_dqn.py \
--exp_name xxx \
--env_name LunarLander-Customizable \
--env_rew_weights a b c d e \
--double_q --seed 1
```
Here the first four values in front of `env_rew_weights` correspond to
intermediate rewards, and the last value corresponds to the final reward. 
Values can be float type.

You might want to choose a distinguishing `exp_name` 
since it is how different modules find previously saved runs.

### Default rewards
To run DQN with default rewards, you just need to set `env_rew_weights` to all one:
```shell
python cs285/scripts/run_dqn.py \
--exp_name xxx_default \
--env_name LunarLander-Customizable \
--env_rew_weights 1 1 1 1 1 \
--double_q --seed 1
```
I recommend using postfix `_default` for `exp_name` 
when you run on an environment with default rewards.

### Sparse rewards 
To run DQN with sparse rewards set `env_rew_weights` 
last input to one and others to zero:
```shell
python cs285/scripts/run_dqn.py \
--exp_name xxx_sparse \
--env_name LunarLander-Customizable \
--env_rew_weights 0 0 0 0 1 \
--double_q --seed 1
```
I recommend using postfix `_sparse` for `exp_name` 
when you run on an environment with default rewards.

### Vector rewards
Some algorithms require access to the vector of rewards. For LunarLander, set `env_name` to `LunarLander-MultiReward`
and `LunarLander-MultiInterReward` to get all or intermediate rewards vector from the environment.

### Offline env
TODO

## Training (phase 1)

### IndependentDQN
To run IndependentDQN, you should first run a couple of DQNs with a unique prefix in `exp_name`.
`run_dqn` by default trains a DQN as mentioned above. 
Look at [run_dqn.py](cs285/scripts/run_dqn.py) for further available options.

### MDQN
Three types of MDQN are available. All types require an environment with vector reward.
#### OptimisticMDQN
Set `optimistic_mdqn` with optional parameter `w_bound` (default=0.5). 
OptimisticMDQN uses [LinearlyWeightedArgMaxPolicy](cs285/policies/linearly_weighted_argmax_policy.py)
where weights will be drawn from unif(1, 1 + w_bound)
```shell
python cs285/scripts/run_dqn.py \
--exp_name xxx_omdqn \
--env_name LunarLander-MultiInterReward \
--optimistic_mdqn \
--w_bound y \
--double_q --seed 1
```
#### DiverseMDQN
Set `diverse_mdqn` with additional optional parameter `w_bound` similar to OptimisticMDQN. 
```shell
python cs285/scripts/run_dqn.py \
--exp_name xxx_dmdqn \
--env_name LunarLander-MultiInterReward \
--diverse_mdqn \
--w_bound y \
--double_q --seed 1
```
#### ConsistentMDQN
Set `consistent_mdqn` with two additional optional parameters `w_bound` and `consistency_alpha` (default=0, to be non-negative).
ConsistentMDQN uses [LinearlyWeightedSoftmaxPolicy](cs285/policies/linearly_weighted_argmax_policy.py)
where $\alpha$ determines the hardness of softmax. 
```shell
python cs285/scripts/run_dqn.py \
--exp_name xxx_cmdqn \
--env_name LunarLander-MultiInterReward \
--consistent_mdqn \
--w_bound y \
--consistency_alpha z \
--double_q --seed 1
```

### ExtendedMDQN
Tow types of EMDQN is available. Both type require an environment with vector reward.
#### DiverseEMDQN
Set `diverse_emdqn` with two parameters `ex_dim` (positive integer, default=1) and `w_bound` (optional). 
`ex_dim` is the dimension of the new axis added to Q function. So, Q function is (ac_dim x re_dim x ex_dim) dimensional in EMDQN.
```shell
python cs285/scripts/run_dqn.py \
--exp_name xxx_demdqn \
--env_name LunarLander-MultiInterReward \
--diverse_emdqn \
--ex_dim e \
--w_bound y \
--double_q --seed 1
```
#### ConsistentEMDQN
Set `consistent_emdqn` with three parameters `ex_dim`, `w_bound`, and `consistency_alpha`.
```shell
cs285/scripts/run_dqn.py \
--exp_name xxx_cemdqn \
--env_name LunarLander-MultiInterReward \
--consistent_emdqn \
--ex_dim e \
--w_bound y \
--consistency_alpha z \
--double_q --seed 1
```


## Training (phase 2)

### IndependentDQN
To run IndependentDQN, set `prune_with_idqn`. You should have already run a couple of DQNs and 
specify their unique prefix as `pruning_file_prefix` argument.
Optionally you can use the `pruning_eps` argument (default=0).
- The program will automatically look in the [data](data) folder for all saved runs with
this prefix and load their critics. The loaded critics will be used in [IDQNPruner](cs285/pruners/independent_dqns_pruner.py)
to prune actions at each state. 
- `pruning_eps` determines how strict is Pareto optimality criterion. 
The larger $\epsilon$, the smaller will be the size of Pareto optimal sets 
(look at [ParetoOptimalPruner](cs285/pruners/primary_pruner.py)).

For example, in LunarLander with sparse rewards:
```shell
python cs285/scripts/run_dqn.py \
--exp_name xxx_pruned_idqn_sparse \
--env_name LunarLander-Customizable \
--env_rew_weights 0 0 0 0 1 \
--prune_with_idqn \
--pruning_file_prefix yyy \
--pruning_eps z \
--double_q --seed 1
```

### MDQN
To run DQN with action sets pruned by MDQN, regardless of type of the MDQN, set `prune_with_mdqn`
and give the trained MDQN's unique file prefix as the `pruning_file_prefix` argument. 
You can optionally set `pruning_n_draw` with a positive integer. 
[MDQNPruner](cs285/pruners/dqn_pruner.py) draws different weightings `pruning_n_draw` times and
returns actions which are optimal for at least one of the realized weighting.
```shell
python cs285/scripts/run_dqn.py \
--exp_name xxx_pruned_cmdqn_sparse \
--env_name LunarLander-Customizable \
--env_rew_weights 0 0 0 0 1 \
--prune_with_mdqn \
--pruning_file_prefix yyy \
--pruning_n_draw n \
--double_q --seed 1
```

### ExtendedMDQN
To run DQN with action sets pruned by EMDQN, regardless of type of the EMDQN, set `prune_with_emdqn`
and give the trained EMDQN's unique file prefix as the `pruning_file_prefix` argument.
```shell
python cs285/scripts/run_dqn.py \
--exp_name xxx_pruned_cemdqn_sparse \
--env_name LunarLander-Customizable \
--env_rew_weights 0 0 0 0 1 \
--prune_with_emdqn \
--pruning_file_prefix yyy \
--pruning_n_draw n \
--double_q --seed 1
```

## Evaluation
`run_eval_pruning` extracts pruned action sets in trajectories of an optimal agent. 
Folder [figs](figs) has sample evaluation outputs for the sample data.
In the following there are sample commands for different methods.

### IndependentDQN
```shell
python cs285/scripts/run_eval_pruning.py \
--exp_name xxx_idqn_eval \
--env_name LunarLander-Customizable \
--env_rew_weights 0 0 0 0 1 \
--opt_file_prefix yyy \
--prune_with_idqn \
--pruning_file_prefix zzz \
--pruning_eps w \
--seed 1
```

### MDQN
```shell
python cs285/scripts/run_eval_pruning.py \
--exp_name xxx_cmdqn_eval \
--env_name LunarLander-Customizable \
--env_rew_weights 0 0 0 0 1 \
--opt_file_prefix yyy \
--prune_with_mdqn \
--pruning_file_prefix zzz \
--pruning_n_draw n \
--seed 1
```

### ExtendedMDQN
```shell
python cs285/scripts/run_eval_pruning.py \
--exp_name xxx_cemdqn_eval \
--env_name LunarLander-Customizable \
--env_rew_weights 0 0 0 0 1 \
--opt_file_prefix yyy \
--prune_with_emdqn \
--pruning_file_prefix zzz \
--pruning_n_draw n \
--seed 1
```


## Post-process and visualization
Use [post_process_training_logs](cs285/scripts/post_process_training_logs.py) to visualize training logs.

Use [post_process_eval_pareto_opt_dqn](cs285/scripts/post_process_eval_pruning.py)
to read the results of pruning evaluation.

## Sample data
Folder [data](data) has some sample runs with `exp_name=p4`. 
Do not push your data unless we want them for the final report.

### MIMIC offline experiments

This implements offline DQN training for the MIMIC data the reward weights each refer to a different reward. The order of the rewards is as follows

'sparse_90d_rew': Reward of 100 if patient survived for 90 days else -100
'Reward_matrix_paper': Paper codebase creates a Reward matrix, rewards her correspond to application of Reward matrix to transition dataset (do not use this)
'Reward_SOFA_1_continous': Reward corresponds to negative of one-period change in SOFA score (t to t+1)
'Reward_SOFA_1_binary': Reward corresponds to -1 if SOFA score increased (t to t+1)
'Reward_SOFA_2_continous': Reward corresponds to negative of two-period change in SOFA score (t to t+2)
'Reward_SOFA_2_binary': Reward corresponds to -1 if SOFA score increased (t to t+2)
'Reward_SOFA_change2_binary': Reward corresponds to -1 if SOFA score increased by at least 2 (t to t+1)
'Reward_lac_1_continous': Reward corresponds to negative of one-period change in lactate levels (t to t+1)
'Reward_lac_1_binary': Reward corresponds to -1 if lactate levels increased (t to t+1)
'Reward_lac_2_continous': Reward corresponds to negative of two-period change in lactate levels (t to t+2)
'Reward_lac_2_binary': Reward corresponds to -1 if lactate levels increased (t to t+2)

So for instance --env_rew_weights 1 0 0 0 0 0 0 0 0 0 0 creates the sparse reward baseline DQN model.

```shell
python cs285/scripts/run_dqn.py \
--exp_name ignore_default \
--env_name MIMIC \
--env_rew_weights 1 0 0 0 0 0 0 0 0 0 0 \
--double_q \
--seed 1 \
--offline \
--no_weights_in_path --buffer_path './Replay_buffer_extraction/Encoded_paths_all_rewards.pkl'
```

### To run Pruned DQN

```shell
python cs285/scripts/run_dqn.py \
--exp_name pDQNvdl_30 \
--env_name MIMIC \
--pruning_file_prefix MIMICvdl_ \
--pruning_eps 0.3 \
--env_rew_weights 1 0 0 0 0 0 0 0 0 0 0 \
--double_q --seed 1 \
--scalar_log_freq 2000 --params_log_freq 2000 \
--offline --no_weights_in_path \
--buffer_path './Replay_buffer_extraction/Encoded_paths3_all_rewards.pkl'
```

python cs285/scripts/run_dqn.py --exp_name off_pDQN{eps} --env_name MIMIC --pruning_file_prefix MIMIC_ --pruning_eps {eps} --env_rew_weights 1 0 0 0 0 0 0 0 0 0 0 --double_q --seed 1 --no_gpu --offline --no_weights_in_path --buffer_path './Replay_buffer_extraction/Encoded_paths_all_rewards.pkl

### To run Pruned DQN Evaluation

```shell
python cs285/scripts/run_eval_pruning.py \
--exp_name pDQNvdl30_eval \
--env_name MIMIC \
--pruning_file_prefix MIMICvdl_ \
--pruning_eps 0.3 \
--env_rew_weights 1 0 0 0 0 0 0 0 0 0 0 \
--seed 1 \
--offline \
--no_weights_in_path --buffer_path '../../Replay_buffer_extraction/Encoded_paths3_all_rewards.pkl' 
```

python run_eval_pareto_opt_dqn.py --exp_name pDQN_30_eval --env_name MIMIC --pruning_file_prefix MIMIC_[Sbl][aO] --pruning_eps 0.3 --env_rew_weights 1 0 0 0 0 0 0 0 0 0 0 --seed 1 --offline --no_weights_in_path --buffer_path  --no_gpu

python run_eval_pareto_opt_dqn.py \
--exp_name pDQN_0_eval \
--env_name MIMIC \
--pruning_file_prefix MIMIC_[Sbl][aO] \
--pruning_eps 0.0 \
--env_rew_weights 1 0 0 0 0 0 0 0 0 0 0 \
--seed 1 \
--offline \
--no_weights_in_path --buffer_path '../../Replay_buffer_extraction/Encoded_paths_all_rewards.pkl' \
--no_gpu


## New approach to create eval files

This command line arguments takes the trained pCQL model and creates an actions file which include the pareto_actions per state, physician actions, and the actions suggested by the respective policy as well as the estimated Q-values.

```
python cs285/scripts/run_eval_pruning.py \
--exp_name test_eval --env_name MIMIC \ --phase_2_critic_file_prefix pCQLv4_10 \
--pruning_file_prefix MIMICCQLv4_ --pruning_eps 0.1 \
--prune_with_icql --env_rew_weights 1 0 0 0 0 0 0 0 0 0 0 \
--seed 1 --offline --no_weights_in_path \ 
--buffer_path './Replay_buffer_extraction/Encoded_paths13_all_rewards.pkl'
```

And for CMDQN

```
python cs285/scripts/run_eval_pruning.py --exp_name 22testcmdqn_eval_cmdqn \
--env_name MIMIC --phase_2_critic_file_prefix v6_offline_pruned_cmdqn \
--pruning_file_prefix v6_offline_cmdqn_alpha --prune_with_mdqn \
--env_rew_weights 1 0 0 0 0 0 0 0 0 0 0 --seed 1 --offline \
--no_weights_in_path \
--buffer_path './Replay_buffer_extraction/Encoded_paths_all_rewards.pkl'
```

From Ali
python cs285/scripts/run_eval_pruning.py --exp_name 22testcmdqn_eval_cmdqn \
--env_name MIMIC --phase_2_critic_file_prefix v6_offline_pruned_cmdqn_alpha1.0_cql0.001_sparse_MIMIC \
--pruning_file_prefix v6_offline_cmdqn_alpha1.0_cql0.001_MIMIC --prune_with_mdqn \
--env_rew_weights 1 0 0 0 0 0 0 0 0 0 0 --seed 1 --offline \
--no_weights_in_path --pruning_n_draw 100 \
--buffer_path './Replay_buffer_extraction/Encoded_paths_all_rewards.pkl'

## Run post-processing and get action distribution plots

The code snippet below assumes that the eval files have been created in a way such that the file includes the actions suggested by the respective pruned policy (see section above). 

```
python cs285/scripts/post_process_eval_pruning.py --prefix pCQLv4*_eval --pruning_file_prefix MIMICCQLv4_ --show --critic_prefix pCQLv4_ --pruned --prune_with_icql --cql --seed 1 --env_rew_weights 1 0 0 0 0 0 0 0 0 0 0 --buffer_path './Replay_buffer_extraction/Encoded_paths13_all_rewards.pkl'
```

And for CMDQ

```
python cs285/scripts/post_process_eval_pruning.py --prefix testcmdqn_eval \
--pruning_file_prefix v6_offline_cmdqn_alpha --show \
--critic_prefix v6_offline_pruned_cmdqn \
--pruned --prune_with_mdqn --mdqn \
--seed 1 --env_rew_weights 1 0 0 0 0 0 0 0 0 0 0 \
--buffer_path './Replay_buffer_extraction/Encoded_paths_all_rewards.pkl'
```

python cs285/scripts/run_eval_pruning.py \
--exp_name test_eval_biom --env_name MIMIC \
--phase_2_critic_file_prefix pCQLv4_10 \
--pruning_file_prefix MIMICCQLv4_ --pruning_eps 0.1 \
--prune_with_icql --env_rew_weights 1 0 0 0 0 0 0 0 0 0 0 \
--seed 1 --offline --no_weights_in_path \
--buffer_path './Replay_buffer_extraction/Paths_all_rewards_raw_obs.pkl'

python cs285/scripts/post_process_eval_pruning.py --prefix test_eval_biom --pruning_file_prefix MIMICCQLv4_ --show --critic_prefix pCQLv4_10 --pruned --prune_with_icql --cql --seed 1 --env_rew_weights 1 0 0 0 0 0 0 0 0 0 0 --buffer_path './Replay_buffer_extraction/Paths_all_rewards_raw_obs_biomarkers.pkl'