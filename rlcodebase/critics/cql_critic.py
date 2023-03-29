from .base_critic import BaseCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn

from ..infrastructure.utils import pytorch_utils as ptu
from rlcodebase.infrastructure.utils.dqn_utils import get_maximizer_from_available_actions
import rlcodebase.policies.argmax_policy as argmax_policy
from rlcodebase.pruners.base_pruner import BasePruner


class CQLCritic(BaseCritic):

    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)

        self.hparams = hparams
        self.env_name = hparams['env_name']
        self.ob_dim = hparams['ob_dim']

        if isinstance(self.ob_dim, int):
            self.input_shape = (self.ob_dim,)
        else:
            self.input_shape = hparams['input_shape']

        self.ac_dim = hparams['ac_dim']
        self.double_q = hparams['double_q']
        self.grad_norm_clipping = hparams['grad_norm_clipping']
        self.gamma = hparams['gamma']

        self.optimizer_spec = optimizer_spec
        network_initializer = hparams['q_func']
        self.q_net = network_initializer(self.ob_dim, self.ac_dim)
        self.q_net_target = network_initializer(self.ob_dim, self.ac_dim)
        self.optimizer = self.optimizer_spec.constructor(
            self.q_net.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.loss = nn.MSELoss()
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)
        self.cql_alpha = hparams['cql_alpha']

    def get_actor_class(self):
        return argmax_policy.ArgMaxPolicy

    def dqn_loss(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        qa_t_values = self.q_net(ob_no)
        q_t_values = torch.gather(qa_t_values, 1, ac_na.unsqueeze(1)).squeeze(1)
        qa_tp1_values = self.q_net_target(next_ob_no)

        next_actions = self.q_net(next_ob_no).argmax(dim=1)
        q_tp1 = torch.gather(qa_tp1_values, 1, next_actions.unsqueeze(1)).squeeze(1)

        target = reward_n + self.gamma * q_tp1 * (1 - terminal_n)
        target = target.detach()
        loss = self.loss(q_t_values, target)

        return loss, qa_t_values, q_t_values

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
            Update the parameters of the critic.
            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories
            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end
            returns:
                nothing
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)

        loss, qa_t_values, q_t_values = self.dqn_loss(
            ob_no, ac_na, next_ob_no, reward_n, terminal_n
            )
        
        # CQL Implementation
        # <DONE>: Implement CQL as described in the pdf and paper
        # Hint: After calculating cql_loss, augment the loss appropriately
        q_t_logsumexp = torch.logsumexp(qa_t_values, dim=1)
        cql_loss = torch.mean(q_t_logsumexp - q_t_values)
        loss = self.cql_alpha * cql_loss + loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        info = {'Training Loss': ptu.to_numpy(loss)}

        # <DONE>: Uncomment these lines after implementing CQL
        info['CQL Loss'] = ptu.to_numpy(cql_loss)
        info['Data q-values'] = ptu.to_numpy(q_t_values).mean()
        info['OOD q-values'] = ptu.to_numpy(q_t_logsumexp).mean()

        return info

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def qa_values(self, obs):
        obs = ptu.from_numpy(obs)
        qa_values = self.q_net(obs)
        return ptu.to_numpy(qa_values)

    def save(self, save_path):
        torch.save(
            {
                'hparams': self.hparams,
                'optimizer_spec': self.optimizer_spec,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'q_net_state_dict': self.q_net.state_dict(),
                'q_net_target_state_dict': self.q_net_target.state_dict()
            }, save_path)

    @classmethod
    def load(cls, load_path):
        checkpoint = torch.load(load_path)

        cql_critic = cls(
            hparams=checkpoint['hparams'],
            optimizer_spec=checkpoint['optimizer_spec']
        )

        cql_critic.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        cql_critic.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        cql_critic.q_net_target.load_state_dict(checkpoint['q_net_target_state_dict'])

        cql_critic.q_net.to(ptu.device)
        cql_critic.q_net_target.to(ptu.device)

        return cql_critic


class PrunedCQLCritic(CQLCritic):

    def __init__(self, hparams, optimizer_spec, action_pruner: BasePruner, **kwargs):
        super().__init__(hparams, optimizer_spec, **kwargs)

        # Pruning
        self.action_pruner = action_pruner

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
            Update the parameters of the critic.
            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end
            returns:
                nothing
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)

        qa_t_values = self.q_net(ob_no)
        q_t_values = torch.gather(qa_t_values, 1, ac_na.unsqueeze(1)).squeeze(1)

        # Compute the Q-values from the target network
        qa_tp1_target_values = self.q_net_target(next_ob_no)

        # Get the pruned action set
        next_available_actions = self.action_pruner.get_list_of_available_actions(ptu.to_numpy(next_ob_no))

        if self.double_q:
            # In double Q-learning, the best action is selected using the Q-network that
            # is being updated, but the Q-value for this action is obtained from the
            # target Q-network.

            qa_tp1_values = self.q_net(next_ob_no)

            ac_tp1 = get_maximizer_from_available_actions(qa_tp1_values, next_available_actions)

        else:
            ac_tp1 = get_maximizer_from_available_actions(qa_tp1_target_values, next_available_actions)

        q_tp1 = torch.gather(qa_tp1_target_values, 1, ac_tp1.unsqueeze(1)).squeeze(1)

        # Compute targets for minimizing Bellman error
        # currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
        target = reward_n + self.gamma * q_tp1 * (1 - terminal_n)
        target = target.detach()

        assert q_t_values.shape == target.shape
        loss = self.loss(q_t_values, target)

        # Here comes the CQL part
        q_t_logsumexp = torch.logsumexp(qa_t_values, dim=1)
        cql_loss = torch.mean(q_t_logsumexp - q_t_values)
        loss = self.cql_alpha * cql_loss + loss

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        self.learning_rate_scheduler.step()

        info = {'Training Loss': ptu.to_numpy(loss)}

        info['CQL Loss'] = ptu.to_numpy(cql_loss)
        info['Data q-values'] = ptu.to_numpy(q_t_values).mean()
        info['OOD q-values'] = ptu.to_numpy(q_t_logsumexp).mean()

        return info
