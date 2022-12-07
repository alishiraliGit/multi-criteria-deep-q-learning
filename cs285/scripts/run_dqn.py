import sys
import os
import time
import glob
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from cs285.infrastructure.rl_trainer import RLTrainer
from cs285.agents.dqn_agent import DQNAgent
from cs285.agents.pareto_opt_agent import LoadedParetoOptDQNAgent
from cs285.infrastructure.dqn_utils import get_env_kwargs


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
    parser.add_argument('--pruning_eps', type=float, default=0., help='Look at ParetoOptimalPolicy.')

    # System
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)

    # Logging
    parser.add_argument('--scalar_log_freq', type=int, default=int(1e4))
    parser.add_argument('--video_log_freq', type=int, default=-1)
    parser.add_argument('--params_log_freq', type=int, default=int(1e4))  # Saves the trained networks

    args = parser.parse_args()

    # Convert to dictionary
    params = vars(args)

    params['train_batch_size'] = params['batch_size']  # Ensure compatibility

    # Decision booleans
    customize_rew = False if params['env_rew_weights'] is None else True
    prune = False if params['pruning_file_prefix'] is None else True

    ##################################
    # Create directory for logging
    ##################################
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    if customize_rew:
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
    if prune:
        pruning_folder_paths = glob.glob(os.path.join(data_path, params['pruning_file_prefix'] + '*'))
        pruning_file_paths = [os.path.join(f, 'dqn_agent.pt') for f in pruning_folder_paths]
        pruning_agent = LoadedParetoOptDQNAgent(file_paths=pruning_file_paths, pruning_eps=params['pruning_eps'])
        params['action_pruner'] = pruning_agent.actor

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
