import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from rlcodebase.eval.rl_evaluator import RLEvaluator
from rlcodebase.eval import metrics
from rlcodebase.configs import get_env_kwargs
from rlcodebase.infrastructure.utils import pytorch_utils as ptu
from rlcodebase.infrastructure.utils.dqn_utils import locate_saved_dqn_agent_from_prefix
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
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--env_rew_weights', type=float, nargs='*', default=None)

    # Sizes
    parser.add_argument('--eval_batch_size', type=int, default=1000)
    parser.add_argument('--num_traj', type=int, default=100)

    # Q-learning params
    parser.add_argument('--gamma', type=float, default=1.0)

    # Pruning
    parser.add_argument('--pruning_file_prefix', type=str, default=None)
    parser.add_argument('--prune_with_idqn', action='store_true')
    parser.add_argument('--prune_with_mdqn', action='store_true')
    parser.add_argument('--prune_with_emdqn', action='store_true')
    parser.add_argument('--pruning_eps', type=float, default=0., help='Look at pareto_opt_pruner.')
    parser.add_argument('--pruning_n_draw', type=int, default=20, help='Look at random_pruner.')

    # Path to agents
    parser.add_argument('--opt_file_prefix', type=str, default=None,
                        help='This policy will be used to interact with the env in the off-policy or online settings.')
    parser.add_argument('--phase_2_file_prefix', type=str, default=None,
                        help='This policy will not be used to interact with the environment.')

    # Offline RL params
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--buffer_path', type=str, default=None)

    # System
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)

    # Logging
    parser.add_argument('--log_freq', type=int, default=1)
    parser.add_argument('--safe', action='store_true')
    parser.add_argument('--no_weights_in_path', action='store_true')

    # Output
    parser.add_argument('--ignore_metrics', action='store_true',
                        help='Ignore computing computationally expensive eval metrics.')

    ##################################
    # Preprocess inputs
    ##################################
    args = parser.parse_args()

    # Convert to dictionary
    params = vars(args)

    # Define decision booleans
    customize_rew = False if params['env_rew_weights'] is None else True

    prune_with_idqn = params['prune_with_idqn']
    prune_with_mdqn = params['prune_with_mdqn']
    prune_with_emdqn = params['prune_with_emdqn']

    offline = params['offline']

    ignore_metrics = params['ignore_metrics']

    # Assert inputs
    if offline and params['buffer_path'] is None:
        raise Exception('Please provide a buffer_path to enable offline learning.')

    assert sum([prune_with_idqn, prune_with_mdqn, prune_with_emdqn]) == 1

    assert offline or (params['opt_file_prefix'] is not None)

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
    os.makedirs(data_path, exist_ok=True)

    # Intentionally don't record eval time
    if customize_rew and not params['no_weights_in_path']:
        logdir = args.exp_name + '_' + args.env_name + '-'.join([str(w) for w in params['env_rew_weights']])
    else:
        logdir = args.exp_name + '_' + args.env_name

    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    os.makedirs(logdir, exist_ok=not params['safe'])

    print(f'\n\n\nLOGGING TO: {logdir} \n\n\n')

    ##################################
    # Get env specific arguments
    ##################################
    env_args = get_env_kwargs(params['env_name'])

    for k, v in env_args.items():
        # Don't overwrite the input arguments
        if k not in params:
            params[k] = v

    ##################################
    # Load saved models
    ##################################
    if prune_with_idqn:
        pruning_file_paths = \
            locate_saved_dqn_agent_from_prefix(data_path, params['pruning_file_prefix'], is_unique=False)
        eval_pruner = IDQNPruner(pruning_eps=params['pruning_eps'], saved_dqn_critics_paths=pruning_file_paths)

    elif prune_with_mdqn:
        pruning_file_path = locate_saved_dqn_agent_from_prefix(data_path, params['pruning_file_prefix'])
        eval_pruner = MDQNPruner(n_draw=params['pruning_n_draw'], file_path=pruning_file_path)

    elif prune_with_emdqn:
        pruning_file_path = locate_saved_dqn_agent_from_prefix(data_path, params['pruning_file_prefix'])
        eval_pruner = ExtendedMDQNPruner(n_draw=params['pruning_n_draw'], file_path=pruning_file_path)

    else:
        raise Exception('Pruning method is not identified.')

    # Load phase 2 agent as the eval critic and policy (if provided)
    eval_policy = None
    eval_critic = None
    if params['phase_2_file_prefix'] is not None:
        phase_2_file_path = locate_saved_dqn_agent_from_prefix(data_path, params['phase_2_file_prefix'])

        # PrunedDQNCritic can't be directly loaded b/c its action_pruner is not saved
        eval_critic = DQNCritic.load(phase_2_file_path)
        eval_policy = PrunedArgMaxPolicy(critic=eval_critic, action_pruner=eval_pruner)

    # Load the optimal policy (which will be interacting with the environment)
    opt_policy = None
    if not offline:
        opt_file_path = locate_saved_dqn_agent_from_prefix(data_path, params['opt_file_prefix'])
        opt_policy = LoadedDQNAgent(file_path=opt_file_path).actor

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
    )


if __name__ == '__main__':
    main()
