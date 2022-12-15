import sys
import os
import time
import glob
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from cs285.infrastructure.rl_evaluator import RLEvaluator
from cs285.agents.dqn_agent import LoadedDQNAgent
from cs285.agents.pareto_opt_agent import LoadedParetoOptDQNAgent, LoadedParetoOptMDQNAgent, \
    LoadedParetoOptExtendedMDQNAgent
from cs285.infrastructure.dqn_utils import get_env_kwargs
from cs285.infrastructure import pytorch_util as ptu


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

    # Path to saved models
    parser.add_argument('--pruning_file_prefix', type=str, required=True)
    parser.add_argument('--opt_file_prefix', type=str) #This is only required for the gym-based evaluation

    # Pruning
    parser.add_argument('--pruning_eps', type=float, default=0., help='Look at pareto_opt_policy.')

    # MDQN
    parser.add_argument('--mdqn', action='store_true')

    # EMDQN
    parser.add_argument('--emdqn', action='store_true')

    # System
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)

    #offline RL?
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--buffer_path', type=str, default=None)

    # Data path formatting
    parser.add_argument('--no_weights_in_path', action='store_true')

    args = parser.parse_args()

    # Convert to dictionary
    params = vars(args)

    # Decision booleans
    customize_rew = False if params['env_rew_weights'] is None else True

    mdqn = params['mdqn']
    emdqn = params['emdqn']

    ##################################
    # Create directory for logging
    ##################################
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    if customize_rew:
        if params['no_weights_in_path']:
            logdir = args.exp_name + '_' + args.env_name \
                    + '_' + time.strftime('%d-%m-%Y_%H-%M-%S')
        else:
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
    env_args = get_env_kwargs(params['env_name']) #we have a MIMIC environment for offline RL

    for k, v in env_args.items():
        # Don't overwrite the input arguments
        if k not in params:
            params[k] = v

    ##################################
    # Load saved models
    ##################################
    # set device
    ptu.init_gpu(
        use_gpu=not params['no_gpu'],
        gpu_id=params['which_gpu']
    )

    if mdqn or emdqn:
        if mdqn:
            pruning_folder_paths = glob.glob(os.path.join(data_path, params['pruning_file_prefix'] + '*'))
            assert len(pruning_folder_paths) == 1
            pruning_file_path = os.path.join(pruning_folder_paths[0], 'dqn_agent.pt')
            if mdqn:
                pruning_agent = LoadedParetoOptMDQNAgent(file_path=pruning_file_path,
                                                         pruning_eps=params['pruning_eps'])
            else:
                pruning_agent = LoadedParetoOptExtendedMDQNAgent(file_path=pruning_file_path,
                                                             pruning_eps=params['pruning_eps'])

    else:
        pruning_folder_paths = glob.glob(os.path.join(data_path, params['pruning_file_prefix'] + '*'))
        pruning_file_paths = [os.path.join(f, 'dqn_agent.pt') for f in pruning_folder_paths]
        pruning_agent = LoadedParetoOptDQNAgent(file_paths=pruning_file_paths, pruning_eps=params['pruning_eps'])

    #Lets skip this for offline RL
    if not params['offline']:
        opt_folder_path = glob.glob(os.path.join(data_path, params['opt_file_prefix'] + '*'))[0]
        opt_file_path = os.path.join(opt_folder_path, 'dqn_agent.pt')
        opt_agent = LoadedDQNAgent(file_path=opt_file_path)

    ##################################
    # Run Q-learning
    ##################################
    params['agent_params'] = params

    rl_evaluator = RLEvaluator(params)

    if not params['offline']:
        rl_evaluator.run_evaluation_loop(
            params['num_traj'],
            collect_policy=opt_agent.actor,
            eval_policy=pruning_agent.actor,
            buffer_path=params['buffer_path']
        )
    else:
        rl_evaluator.run_evaluation_loop(
            params['num_traj'],
            collect_policy=None,
            eval_policy=pruning_agent.actor,
            buffer_path=params['buffer_path']
        )


if __name__ == '__main__':
    main()
