## Setup
I recommend creating a new virtual environment for this project and
install the requirements directly from [requirements.txt](requirements.txt).

## Training

### Customized rewards
You can run DQN on LunarLander with customized rewards by specifying
reward weights in the `--env_rew_weights` argument:
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
since it is how a [PrunedDQN](cs285/critics/dqn_critic.py) finds previously saved runs.

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

### PrunedDQN
To run DQN with pruned actions, you need to specify the `pruning_file_prefix` argument and
optionally `pruning_eps` argument (default=0). The larger `pruning_eps`, the smaller will be the size of Pareto optimal sets 
(look at [ParetoOptimalPolicy](cs285/policies/pareto_opt_policy.py)). 
The program will automatically look in the [data](data) folder for all saved runs with
this prefix and load their critics. 
The loaded critics will be used in [ParetoOptimalAgent](cs285/agents/pareto_opt_agent.py)
to prune actions at each state. 

For example, in LunarLander with sparse rewards:
```shell
python cs285/scripts/run_dqn.py \
--exp_name xxx_pruned_sparse \
--env_name LunarLander-Customizable \
--pruning_file_prefix xxx_LunarLander-Customizable \
--pruning_eps yyy \
--env_rew_weights 0 0 0 0 1 \
--double_q --seed 1
```
I recommend using postfix `_pruned` for `exp_name` 
when you run PrunedDQN.


## Evaluation
To extract pruned action sets in trajectories of an optimal agent, run:
```shell
python cs285/scripts/run_eval_pareto_opt_dqn.py \
--exp_name xxx_eval \
--env_name LunarLander-Customizable \
--pruning_file_prefix xxx_LunarLander-Customizable \
--pruning_eps yyy \
--opt_file_prefix xxx_opt_LunarLander-Customizable \
--env_rew_weights 0 0 0 0 1 \
--seed 1
```
Folder [figs](figs) has sample evaluation outputs for the sample data.

## Post-process and visualization
Use [post_process_training_logs](cs285/scripts/post_process_training_logs.py) to visualize training logs.

Use [post_process_eval_pareto_opt_dqn](cs285/scripts/post_process_eval_pareto_opt_dqn.py)
to read the results of Pareto optimality evaluation.

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