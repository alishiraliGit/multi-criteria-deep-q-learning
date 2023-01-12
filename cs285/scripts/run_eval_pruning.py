import sys
import os
import time
import glob
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from cs285.infrastructure.rl_evaluator import RLEvaluator
from cs285.pruners.cql_pruner import ICQLPruner
from cs285.infrastructure.dqn_utils import get_env_kwargs
from cs285.infrastructure import pytorch_util as ptu
from cs285.agents.dqn_agent import LoadedDQNAgent
from cs285.pruners.independent_dqns_pruner import IDQNPruner
from cs285.pruners.dqn_pruner import MDQNPruner, ExtendedMDQNPruner
from cs285.critics.cql_critic import CQLCritic, PrunedCQLCritic
from cs285.critics.dqn_critic import DQNCritic, MDQNCritic, ExtendedMDQNCritic, PrunedDQNCritic


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
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--num_traj', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=1.0)

    # Path to optimal agent
    parser.add_argument('--opt_file_prefix', type=str)  # This is only required for the gym-based evaluation

    # Path to pruning critic trained in run_dqn script
    parser.add_argument('--trained_pruning_critic', type=str, default=None)

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

    # Offline RL?
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--buffer_path', type=str, default=None)

    # CQL
    parser.add_argument('--cql', action='store_true')
    parser.add_argument('--cql_alpha', type=float, default=0.2, help='Higher values indicated stronger OOD penalty.')

    # Data path formatting
    parser.add_argument('--no_weights_in_path', action='store_true')

    args = parser.parse_args()

    # Convert to dictionary
    params = vars(args)

    # Decision booleans
    customize_rew = False if params['env_rew_weights'] is None else True

    prune_with_idqn = params['prune_with_idqn']
    prune_with_mdqn = params['prune_with_mdqn']
    prune_with_emdqn = params['prune_with_emdqn']
    cql = params['cql']
    assert sum([prune_with_idqn, prune_with_mdqn, prune_with_emdqn, cql]) == 1

    

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
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

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

    elif cql:
        pruning_folder_paths = glob.glob(os.path.join(data_path, params['pruning_file_prefix'] + '*'))
        pruning_file_paths = [os.path.join(f, 'dqn_agent.pt') for f in pruning_folder_paths]
        pruner = ICQLPruner(file_paths=pruning_file_paths, pruning_eps=params['pruning_eps'])
    
    
    if params['trained_pruning_critic'] != None:
        pruning_critic = None
        pruning_folder_paths = glob.glob(os.path.join(data_path, params['trained_pruning_critic'] + '*'))
        critic_file_path = [os.path.join(f, 'dqn_agent.pt') for f in pruning_folder_paths]

        print(critic_file_path[0])
        
        if cql:
            pruning_critic = CQLCritic.load(critic_file_path[0])
            #pruning_critic = PrunedCQLCritic.load(critic_file_path[0])
        else:
            pruning_critic = DQNCritic.load(critic_file_path[0])
            #pruning_critic = PrunedDQNCritic.load(critic_file_path[0])

    # Skip this for offline RL
    if params['offline']:
        opt_actor = None
    else:
        opt_folder_path = glob.glob(os.path.join(data_path, params['opt_file_prefix'] + '*'))[0]
        opt_file_path = os.path.join(opt_folder_path, 'dqn_agent.pt')
        opt_agent = LoadedDQNAgent(file_path=opt_file_path)
        opt_actor = opt_agent.actor

    ##################################
    # Run Q-learning
    ##################################
    params['agent_params'] = params

    rl_evaluator = RLEvaluator(params)

    rl_evaluator.run_evaluation_loop(
        params['num_traj'],
        opt_policy=opt_actor,
        eval_pruner=pruner,
        buffer_path=params['buffer_path'],
        pruning_critic=pruning_critic
    )


if __name__ == '__main__':
    main()