import sys
import os
import glob
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from rlcodebase.eval.rl_evaluator import RLEvaluator
from rlcodebase.eval import metrics
from rlcodebase.configs import get_env_kwargs
from rlcodebase.infrastructure.utils import pytorch_utils as ptu
from rlcodebase.agents.dqn_agent import LoadedDQNAgent


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

    # Path to baseline critic
    parser.add_argument('--baseline_file_prefix', type=str, default=None)

    # System
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)

    # Logging
    parser.add_argument('--log_freq', type=int, default=1)

    # Offline RL?
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--buffer_path', type=str, default=None)

    args = parser.parse_args()

    # Convert to dictionary
    params = vars(args)

    # Decision booleans
    offline = params['offline']

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
    env_args = get_env_kwargs(params['env_name'])

    for k, v in env_args.items():
        # Don't overwrite the input arguments
        if k not in params:
            params[k] = v

    ##################################
    # Load saved models
    ##################################
    eval_folder_path = glob.glob(os.path.join(data_path, params['baseline_file_prefix'] + '*'))[0]
    eval_file_path = os.path.join(eval_folder_path, 'dqn_agent.pt')
    eval_agent = LoadedDQNAgent(file_path=eval_file_path)
    eval_policy = eval_agent.actor
    eval_critic = eval_agent.critic

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

    default_metrics = metrics.get_default_metrics() if not offline else metrics.get_default_offline_metrics()

    rl_evaluator.run_evaluation_loop(
        params['num_traj'],
        opt_policy=opt_policy,
        eval_policy=eval_policy,
        eval_critic=eval_critic,
        eval_pruner=None,
        eval_metrics=default_metrics
    )


if __name__ == '__main__':
    main()
