import sys
import os
import time
import glob
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from cs285.infrastructure.rl_trainer import RLTrainer
from cs285.pruners.cql_pruner import ICQLPruner
from cs285.infrastructure.dqn_utils import get_env_kwargs
from cs285.infrastructure import pytorch_util as ptu
from cs285.agents.dqn_agent import DQNAgent
from cs285.pruners.independent_dqns_pruner import IDQNPruner
from cs285.pruners.dqn_pruner import MDQNPruner, ExtendedMDQNPruner


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
    parser.add_argument('--eval_batch_size', type=int, default=1000)

    # Update frequencies
    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--num_critic_updates_per_agent_update', type=int, default=1)

    # Q-learning params
    parser.add_argument('--double_q', action='store_true')

    # Pruning
    parser.add_argument('--pruning_file_prefix', type=str, default=None)
    parser.add_argument('--prune_with_idqn', action='store_true')
    parser.add_argument('--prune_with_mdqn', action='store_true')
    parser.add_argument('--prune_with_emdqn', action='store_true')
    parser.add_argument('--pruning_eps', type=float, default=0., help='Look at pareto_opt_pruner.')
    parser.add_argument('--pruning_n_draw', type=int, default=20, help='Look at random_pruner.')

    # MDQN
    parser.add_argument('--optimistic_mdqn', action='store_true')
    parser.add_argument('--diverse_mdqn', action='store_true')
    parser.add_argument('--consistent_mdqn', action='store_true')

    # EMDQN
    parser.add_argument('--diverse_emdqn', action='store_true')
    parser.add_argument('--consistent_emdqn', action='store_true')
    parser.add_argument('--ex_dim', type=int, default=1)

    parser.add_argument('--consistency_alpha', type=float, default=1, help='Look at MDQN in dqn_critic.')
    parser.add_argument('--w_bound', type=float, default=0.5, help='Look at draw_w in linearly_weighted_argmax_policy.')

    # CQL
    parser.add_argument('--add_cql_loss', action='store_true', help='Adds CQL loss to MDQN and EMDQN.')
    parser.add_argument('--cql', action='store_true')
    parser.add_argument('--cql_alpha', type=float, default=0.2, help='Higher values indicated stronger OOD penalty.')

    # System
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)

    # Logging
    parser.add_argument('--scalar_log_freq', type=int, default=int(5e3))
    parser.add_argument('--video_log_freq', type=int, default=-1)
    parser.add_argument('--params_log_freq', type=int, default=int(5e3))  # Saves the trained networks

    # Offline learning params
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--buffer_path', type=str, default=None)

    # Data path formatting
    parser.add_argument('--no_weights_in_path', action='store_true')

    args = parser.parse_args()

    # Convert to dictionary
    params = vars(args)

    params['train_batch_size'] = params['batch_size']  # Ensure compatibility

    # Decision booleans
    customize_rew = False if params['env_rew_weights'] is None else True

    if params['offline'] and params['buffer_path'] is None:
        raise Exception('Please provide a buffer_path to enable offline learning.')

    params['mdqn'] = params['optimistic_mdqn'] or params['diverse_mdqn'] or params['consistent_mdqn']

    params['emdqn'] = params['diverse_emdqn'] or params['consistent_emdqn']

    prune_with_idqn = params['prune_with_idqn']
    prune_with_mdqn = params['prune_with_mdqn']
    prune_with_emdqn = params['prune_with_emdqn']

    params['prune'] = prune_with_idqn or prune_with_mdqn or prune_with_emdqn

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

    if customize_rew and not params['no_weights_in_path']:
        logdir = args.exp_name + '_' + args.env_name \
                + '-'.join([str(w) for w in params['env_rew_weights']]) \
                + '_' + time.strftime('%d-%m-%Y_%H-%M-%S')
    else:
        logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime('%d-%m-%Y_%H-%M-%S')

    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print('\n\n\nLOGGING TO: ', logdir, '\n\n\n')

    ##################################
    # Get env specific arguments
    ##################################
    env_args = get_env_kwargs(params['env_name'])

    for k, v in env_args.items():
        # Don't overwrite the input arguments
        if k not in params:
            params[k] = v

    ##################################
    # Pruning (if requested)
    ##################################
    if params['prune']:
        pruner = None
        pruning_folder_paths = glob.glob(os.path.join(data_path, params['pruning_file_prefix'] + '*'))

        if prune_with_idqn:
            pruning_file_paths = [os.path.join(f, 'dqn_agent.pt') for f in pruning_folder_paths]
            pruner = IDQNPruner(pruning_eps=params['pruning_eps'], saved_dqn_critics_paths=pruning_file_paths)

        elif prune_with_mdqn:
            assert len(pruning_folder_paths) == 1
            pruning_file_path = os.path.join(pruning_folder_paths[0], 'dqn_agent.pt')
            pruner = MDQNPruner(n_draw=params['pruning_n_draw'], file_path=pruning_file_path)

        elif prune_with_emdqn:
            assert len(pruning_folder_paths) == 1
            pruning_file_path = os.path.join(pruning_folder_paths[0], 'dqn_agent.pt')
            pruner = ExtendedMDQNPruner(n_draw=params['pruning_n_draw'], file_path=pruning_file_path)

        elif params['cql']:
            pruning_file_paths = [os.path.join(f, 'dqn_agent.pt') for f in pruning_folder_paths]
            pruner = ICQLPruner(file_paths=pruning_file_paths, pruning_eps=params['pruning_eps'])

        params['action_pruner'] = pruner

    ##################################
    # Run Q-learning
    ##################################
    params['agent_class'] = DQNAgent
    params['agent_params'] = params

    rl_trainer = RLTrainer(params)
    rl_trainer.run_training_loop(
        params['num_timesteps'],
        collect_policy=rl_trainer.agent.actor,
        eval_policy=rl_trainer.agent.actor
    )


if __name__ == '__main__':
    main()
