import os
import time

from cs285.infrastructure.rl_trainer import RLTrainer
from cs285.agents.dqn_agent import DQNAgent
from cs285.infrastructure.dqn_utils import get_env_kwargs


class QTrainer(object):

    def __init__(self, params):
        self.params = params

        env_args = get_env_kwargs(params['env_name'])

        for k, v in env_args.items():
            if k not in self.params:
                self.params[k] = v

        self.params['agent_class'] = DQNAgent
        self.params['train_batch_size'] = params['batch_size']  # Ensure compatibility
        self.params['agent_params'] = self.params

        self.rl_trainer = RLTrainer(self.params)

    def run_training_loop(self):
        self.rl_trainer.run_training_loop(
            self.params['num_timesteps'],
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
            )


class Args:
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, val):
        setattr(self, key, val)

    def __contains__(self, key):
        return hasattr(self, key)

    env_name = 'LunarLander-Customizable'
    exp_name = 'test_customizable_ll'

    ep_len = 200  # Env determines

    # Batches and steps
    batch_size = 32
    eval_batch_size = 1000

    num_agent_train_steps_per_iter = 1

    num_critic_updates_per_agent_update = 1

    # Q-learning parameters
    double_q = True

    # System
    save_params = True
    no_gpu = True
    which_gpu = 0
    seed = 1

    # Logging
    video_log_freq = -1
    scalar_log_freq = 10000


def init():
    # Init params
    args = Args()

    # Create a logging directory
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    args['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    # Init a trainer
    trainer = QTrainer(args)

    return trainer


if __name__ == "__main__":
    trainer_ = init()

    trainer_.run_training_loop()
