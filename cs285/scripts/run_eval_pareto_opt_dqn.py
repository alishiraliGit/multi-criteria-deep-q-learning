import os
import time
import glob

from cs285.infrastructure.rl_evaluator import RLEvaluator
from cs285.infrastructure.dqn_utils import get_env_kwargs
from cs285.agents.dqn_agent import LoadedDQNAgent
from cs285.agents.pareto_opt_agent import LoadedParetoOptDQNAgent

# TODO: Get rid of Args() so pickle can load params!


class ParetoOptQEvaluator(object):

    def __init__(self, params):
        # Update params with env params
        self.params = params

        env_args = get_env_kwargs(params['env_name'])

        for k, v in env_args.items():
            if k not in self.params:
                self.params[k] = v

        # Ensure compatibility
        self.params['train_batch_size'] = params['batch_size']
        self.params['agent_params'] = self.params

        # Load agents
        self.opt_agent = LoadedDQNAgent(self.params['optimal_critic_file_path'])
        self.pareto_opt_agent = LoadedParetoOptDQNAgent(self.params['other_critics_file_paths'])

        # Init RLEvaluator
        self.rl_evaluator = RLEvaluator(self.params)

    def run_evaluation_loop(self):
        return self.rl_evaluator.run_evaluation_loop(
            self.params['num_timesteps'],
            collect_policy=self.opt_agent.actor,
            eval_policy=self.pareto_opt_agent.actor,
            )


class EvalArgs:
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, val):
        setattr(self, key, val)

    def __contains__(self, key):
        return hasattr(self, key)

    env_name = 'LunarLander-Sparse'
    exp_name = 'eval_p0'

    ep_len = 200  # Env determines

    # Batches and steps
    batch_size = 1000
    num_timesteps = 100

    # System
    no_gpu = True
    which_gpu = 0
    seed = 1


def init():
    # Init params
    args = EvalArgs()

    # Create a logging directory
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime('%d-%m-%Y_%H-%M-%S')
    logdir = os.path.join(data_path, logdir)
    args['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    # Find loaded models' paths
    prefix_opt = 'p0_LunarLander-1155'
    folder_path_opt = glob.glob(os.path.join(data_path, prefix_opt + '*'))[0]
    args['optimal_critic_file_path'] = os.path.join(folder_path_opt, 'dqn_agent.pt')

    prefix_others = 'p0_LunarLander'
    folder_paths_others = glob.glob(os.path.join(data_path, prefix_others + '*'))
    if folder_path_opt in folder_paths_others:
        folder_paths_others.remove(folder_path_opt)
    args['other_critics_file_paths'] = [os.path.join(f, 'dqn_agent.pt') for f in folder_paths_others]

    # Init a trainer
    evaluator = ParetoOptQEvaluator(args)

    return evaluator


if __name__ == '__main__':
    evaluator_ = init()

    opt_actions_, pareto_opt_actions_ = evaluator_.run_evaluation_loop()
