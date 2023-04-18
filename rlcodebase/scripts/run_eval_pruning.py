import sys
import os
import glob
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from rlcodebase.eval.rl_evaluator import RLEvaluator
from rlcodebase.eval import metrics
from rlcodebase.configs import get_env_kwargs
from rlcodebase.infrastructure.utils import pytorch_utils as ptu
from rlcodebase.infrastructure.utils.general_utils import escape_bracket_globe
from rlcodebase.agents.dqn_agent import LoadedDQNAgent
from rlcodebase.pruners.independent_dqns_pruner import IDQNPruner
from rlcodebase.pruners.dqn_pruner import MDQNPruner, ExtendedMDQNPruner
from rlcodebase.critics.dqn_critic import DQNCritic
from rlcodebase.policies.argmax_policy import PrunedArgMaxPolicy


def main():
    ##################################
    # Get arguments from input
    ##################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str)

    # Env
    parser.add_argument('--env_name', type=str, default='LunarLander-Customizable')
    parser.add_argument('--env_rew_weights', type=float, nargs='*', default=None)

    # Batch size
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_traj', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=1.0)

    # Update frequencies
    parser.add_argument('--target_update_freq', type=int, default=3000)

    # Q-learning params
    parser.add_argument('--arch_dim', type=int, default=64)

    # Path to optimal agent
    parser.add_argument('--opt_file_prefix', type=str)  # This is only required for the gym-based evaluation

    # Path to phase 2 final critic
    parser.add_argument('--phase_2_critic_file_prefix', type=str, default=None)

    # Pruning
    parser.add_argument('--pruning_file_prefix', type=str, default=None)
    parser.add_argument('--prune_with_idqn', action='store_true')
    parser.add_argument('--prune_with_mdqn', action='store_true')
    parser.add_argument('--prune_with_emdqn', action='store_true')
    parser.add_argument('--pruning_eps', type=float, default=0., help='Look at pareto_opt_pruner.')
    parser.add_argument('--pruning_n_draw', type=int, default=20, help='Look at random_pruner.')

    # System
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)

    # Logging
    parser.add_argument('--log_freq', type=int, default=1)

    # Offline RL?
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--buffer_path', type=str, default=None)

    # Do we need to compute computationally expensive eval metrics?
    parser.add_argument('--ignore_metrics', action='store_true')

    # Do I want the output dict to maintain the trajectory structure?
    parser.add_argument('--maintain_traj', action='store_true')

    args = parser.parse_args()

    # Convert to dictionary
    params = vars(args)

    # Decision booleans
    prune_with_idqn = params['prune_with_idqn']
    prune_with_mdqn = params['prune_with_mdqn']
    prune_with_emdqn = params['prune_with_emdqn']
    assert sum([prune_with_idqn, prune_with_mdqn, prune_with_emdqn]) == 1

    offline = params['offline']

    ignore_metrics = params['ignore_metrics']

    ##################################
    # Set system variables
    ##################################
    # Set device
    ptu.init_gpu(
        use_gpu=not params['no_gpu'],
        gpu_id=params['which_gpu']
    )

    ##################################
    # Create directory for logging
    ##################################
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    # Intentionally don't record eval time
    logdir = args.exp_name + '_' + args.env_name

    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    print('\n\n\nLOGGING TO: ', logdir, '\n\n\n')

    ##################################
    # Get env specific arguments
    ##################################
    env_args = get_env_kwargs(params['env_name'])  # We have a MIMIC environment for offline RL

    for k, v in env_args.items():
        # Don't overwrite the input arguments
        if k not in params:
            params[k] = v

    ##################################
    # Load saved models
    ##################################
    eval_pruner = None
    pruning_folder_paths = glob.glob(os.path.join(data_path, params['pruning_file_prefix'] + '*'))

    if prune_with_idqn:
        pruning_file_paths = [os.path.join(f, 'dqn_agent.pt') for f in pruning_folder_paths]
        eval_pruner = IDQNPruner(pruning_eps=params['pruning_eps'], saved_dqn_critics_paths=pruning_file_paths)

    elif prune_with_mdqn:
        assert len(pruning_folder_paths) == 1, 'found %d files!' % len(pruning_folder_paths)
        pruning_file_path = os.path.join(pruning_folder_paths[0], 'dqn_agent.pt')
        eval_pruner = MDQNPruner(n_draw=params['pruning_n_draw'], file_path=pruning_file_path)

    elif prune_with_emdqn:
        assert len(pruning_folder_paths) == 1, 'found %d files!' % len(pruning_folder_paths)
        pruning_file_path = os.path.join(pruning_folder_paths[0], 'dqn_agent.pt')
        eval_pruner = ExtendedMDQNPruner(n_draw=params['pruning_n_draw'], file_path=pruning_file_path)

    # Load phase 2 critic if provided
    eval_policy = None
    eval_critic = None
    if params['phase_2_critic_file_prefix'] is not None:
        phase_2_critic_file_prefix = escape_bracket_globe(params['phase_2_critic_file_prefix'])
        critic_folder_paths = glob.glob(os.path.join(data_path, phase_2_critic_file_prefix + '*'))
        assert len(critic_folder_paths) == 1, 'found %d files!' % len(critic_folder_paths)
        critic_file_path = os.path.join(critic_folder_paths[0], 'dqn_agent.pt')

        # PrunedDQNCritic can't be directly loaded b/c its action_pruner is not saved
        eval_critic = DQNCritic.load(critic_file_path)
        eval_policy = PrunedArgMaxPolicy(critic=eval_critic, action_pruner=eval_pruner)

    # Skip this for offline RL
    if offline:
        opt_policy = None
    else:
        opt_folder_path = glob.glob(os.path.join(data_path, params['opt_file_prefix'] + '*'))[0]
        opt_file_path = os.path.join(opt_folder_path, 'dqn_agent.pt')
        opt_agent = LoadedDQNAgent(file_path=opt_file_path)
        opt_policy = opt_agent.actor

    ##################################
    # Run Q-learning
    ##################################
    params['agent_params'] = params

    rl_evaluator = RLEvaluator(params)

    default_metrics = metrics.get_default_pruning_metrics()
    if eval_policy is not None:
        default_metrics += metrics.get_default_metrics() if not offline else metrics.get_default_offline_metrics()

    rl_evaluator.run_evaluation_loop(
        params['num_traj'],
        opt_policy=opt_policy,
        eval_policy=eval_policy,
        eval_critic=eval_critic,
        eval_pruner=eval_pruner,
        eval_metrics=default_metrics,
        ignore_metrics=ignore_metrics,
        get_traj_structure=params['maintain_traj']
    )


if __name__ == '__main__':
    main()
