from .base_critic import BaseCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn
import torch.nn.functional as F
import copy

from ..infrastructure.utils import pytorch_utils as ptu
from rlcodebase.infrastructure.utils.dqn_utils import get_maximizer_from_available_actions
import rlcodebase.policies.argmax_policy as argmax_policy
import rlcodebase.policies.stochastic_bcq_policy as bcq_policy
from rlcodebase.pruners.base_pruner import BasePruner


class BCQCritic(BaseCritic):

    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.hparams = hparams

        # Initialize parameters
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

        #Initialize Q functions
        self.q_net = FC_Q(self.ob_dim, self.ac_dim)
        self.q_net_target = copy.deepcopy(self.q_net)

        # Optimization
        self.optimizer_spec = optimizer_spec
        self.optimizer = self.optimizer_spec.constructor(
            self.q_net.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )

        self.loss = nn.SmoothL1Loss()  # AKA Huber loss
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)

        # BCQ threshold
        self.bcq_thres = hparams['bcq_thres']

        # Polyak target update
        self.polyak = hparams['polyak_target_update']
        self.tau = hparams['tau']

    def get_actor_class(self):
        return bcq_policy.BCQPolicy

    def bcq_loss(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):

        # Compute the target Q value
        with torch.no_grad():
            q, imt, i = self.q_net(next_ob_no)
            imt = imt.exp()
            imt = (imt/imt.max(1, keepdim=True)[0] > self.bcq_thres).float()

			# Use large negative number to mask actions from argmax
            next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)

            q, imt, i = self.q_net_target(next_ob_no)
            target_Q = reward_n + terminal_n * self.gamma * q.gather(1, next_action).reshape(-1, 1)
        
        # Get current Q estimate
        current_Qa, imt, i = self.q_net(ob_no)

        current_Q_t = current_Qa.gather(1, ac_na.unsqueeze(1))

        # Compute Q loss
        q_loss = F.smooth_l1_loss(current_Q_t, target_Q)
        i_loss = F.nll_loss(imt, ac_na.reshape(-1))
        loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()

        return loss, current_Qa, current_Q_t
        
    # def dqn_loss(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
    #     qa_t_values = self.q_net(ob_no)
    #     q_t_values = torch.gather(qa_t_values, 1, ac_na.unsqueeze(1)).squeeze(1)
    #     qa_tp1_values = self.q_net_target(next_ob_no)

    #     next_actions = self.q_net(next_ob_no).argmax(dim=1)
    #     q_tp1 = torch.gather(qa_tp1_values, 1, next_actions.unsqueeze(1)).squeeze(1)

    #     target = reward_n + self.gamma * q_tp1 * (1 - terminal_n)
    #     target = target.detach()
    #     loss = self.loss(q_t_values, target)

    #     return loss, qa_t_values, q_t_values

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

        loss, qa_t_values, q_t_values = self.bcq_loss(
            ob_no, ac_na, next_ob_no, reward_n, terminal_n
            )
        
        # CQL Implementation
        # <DONE>: Implement CQL as described in the pdf and paper
        # Hint: After calculating cql_loss, augment the loss appropriately
        # q_t_logsumexp = torch.logsumexp(qa_t_values, dim=1)
        # cql_loss = torch.mean(q_t_logsumexp - q_t_values)
        # loss = self.cql_alpha * cql_loss + loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        info = {'Training Loss': ptu.to_numpy(loss)}

        # <DONE>: Uncomment these lines after implementing CQL
        info['BCQ Loss'] = ptu.to_numpy(loss)
        info['Data q-values'] = ptu.to_numpy(q_t_values).mean()
        #info['OOD q-values'] = ptu.to_numpy(q_t_logsumexp).mean()

        return info

    def update_target_network(self):
        if self.polyak:
             for param, target_param in zip(self.q_net.parameters(), self.q_net_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
             
        else:
            for target_param, param in zip(
                    self.q_net_target.parameters(), self.q_net.parameters()
            ):
                target_param.data.copy_(param.data)

    def qa_values(self, obs):
        obs = ptu.from_numpy(obs)
        q, imt, i = self.q_net(obs)
        imt = imt.exp()
        imt = (imt/imt.max(1, keepdim=True)[0] > self.bcq_thres).float()
        q = imt * q + (1 - imt) * -1e8
        return q, imt, i

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

        bcq_critic = cls(
            hparams=checkpoint['hparams'],
            optimizer_spec=checkpoint['optimizer_spec']
        )

        bcq_critic.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        bcq_critic.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        bcq_critic.q_net_target.load_state_dict(checkpoint['q_net_target_state_dict'])

        bcq_critic.q_net.to(ptu.device)
        bcq_critic.q_net_target.to(ptu.device)

        return bcq_critic

# Used for Box2D / Toy problems
class FC_Q(nn.Module):
	def __init__(self, state_dim, num_actions):
		super(FC_Q, self).__init__()
		self.q1 = nn.Linear(state_dim, 256)
		self.q2 = nn.Linear(256, 256)
		self.q3 = nn.Linear(256, num_actions)

		self.i1 = nn.Linear(state_dim, 256)
		self.i2 = nn.Linear(256, 256)
		self.i3 = nn.Linear(256, num_actions)		


	def forward(self, state):
		q = F.relu(self.q1(state))
		q = F.relu(self.q2(q))

		i = F.relu(self.i1(state))
		i = F.relu(self.i2(i))
		i = self.i3(i)
		return self.q3(q), F.log_softmax(i, dim=1), i
