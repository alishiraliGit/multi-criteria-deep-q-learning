import numpy as np
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn

from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure.dqn_utils import get_maximizer_from_available_actions, gather_by_actions, gather_by_e
from cs285.critics.base_critic import BaseCritic
from cs285.policies.pareto_opt_policy import ParetoOptimalPolicy


class DQNCritic(BaseCritic):

    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)

        self.hparams = hparams

        # Env
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

        # Networks
        network_initializer = hparams['q_func']
        self.q_net = network_initializer(self.ob_dim, self.ac_dim)
        self.q_net_target = network_initializer(self.ob_dim, self.ac_dim)

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
        qa_tp1_values = self.q_net_target(next_ob_no)

        if self.double_q:
            # In double Q-learning, the best action is selected using the Q-network that
            # is being updated, but the Q-value for this action is obtained from the
            # target Q-network.
            ac_tp1 = self.q_net(next_ob_no).argmax(dim=1)
            q_tp1 = torch.gather(qa_tp1_values, 1, ac_tp1.unsqueeze(1)).squeeze(1)
        else:
            q_tp1, _ = qa_tp1_values.max(dim=1)

        # Compute targets for minimizing Bellman error
        # currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
        target = reward_n + self.gamma*q_tp1*(1 - terminal_n)
        target = target.detach()

        assert q_t_values.shape == target.shape
        loss = self.loss(q_t_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        self.learning_rate_scheduler.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def qa_values(self, obs:  np.ndarray) -> np.ndarray:
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

    @staticmethod
    def load(load_path):
        checkpoint = torch.load(load_path)

        dqn_critic = DQNCritic(
            hparams=checkpoint['hparams'],
            optimizer_spec=checkpoint['optimizer_spec']
        )

        dqn_critic.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        dqn_critic.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        dqn_critic.q_net_target.load_state_dict(checkpoint['q_net_target_state_dict'])

        return dqn_critic


class MDQNCritic(BaseCritic):

    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)

        self.hparams = hparams

        # Env
        self.env_name = hparams['env_name']
        self.ob_dim = hparams['ob_dim']

        if isinstance(self.ob_dim, int):
            self.input_shape = (self.ob_dim,)
        else:
            self.input_shape = hparams['input_shape']

        self.ac_dim = hparams['ac_dim']
        self.re_dim = hparams['re_dim']

        self.grad_norm_clipping = hparams['grad_norm_clipping']
        self.gamma = hparams['gamma']

        # Networks
        network_initializer = hparams['q_func']
        self.mq_net = network_initializer(self.ob_dim, self.ac_dim, self.re_dim)
        self.mq_net_target = network_initializer(self.ob_dim, self.ac_dim, self.re_dim)

        # Optimization
        self.optimizer_spec = optimizer_spec
        self.optimizer = self.optimizer_spec.constructor(
            self.mq_net.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )

        self.loss = nn.SmoothL1Loss()  # AKA Huber loss

        self.mq_net.to(ptu.device)
        self.mq_net_target.to(ptu.device)

        # Pruning
        self.eps = hparams['pruning_eps']
        self.optimistic = hparams.get('optimistic_mdqn', False)
        self.uniform_consistent = hparams.get('uniform_consistent_mdqn', False)
        self.consistent = hparams.get('consistent_mdqn', False)
        if np.sum([self.optimistic, self.consistent, self.uniform_consistent]) > 1:
            raise Exception('MDQN type is inconsistently defined!')
        self.alpha = hparams['consistency_alpha']
        self.b = hparams['w_bound']

    def update(self, ob_no, ac_n, next_ob_no, reward_nr, terminal_n):
        """
            Update the parameters of the critic.
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_n = ptu.from_numpy(ac_n).to(torch.long)  # If action space is discrete, ac_n
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_nr = ptu.from_numpy(reward_nr)
        terminal_n = ptu.from_numpy(terminal_n)

        # Get the current Q-values
        qa_t_values_nar = self.mq_net(ob_no)
        q_t_values_nr = gather_by_actions(qa_t_values_nar, ac_n)

        # Compute the Q-values for the next observation
        qa_tp1_values_nar = self.mq_net_target(next_ob_no).detach()

        # Find next actions and Q-values based on MDQN type

        #####################
        # Optimistic
        #####################
        if self.optimistic:
            q_tp1_values_nr, _ = qa_tp1_values_nar.max(dim=1)

        #####################
        # Uniform consistent
        #####################
        # Select the action for uniform consistent
        elif self.uniform_consistent:
            # Draw ws
            w1_nr = torch.rand(q_t_values_nr.shape)*self.b + 1
            w2_nr = torch.rand(q_t_values_nr.shape)*self.b + 1

            # Find the inner-products
            prod_1_n = (q_t_values_nr.detach() * w1_nr).sum(dim=1)
            prod_2_n = (q_t_values_nr.detach() * w2_nr).sum(dim=1)

            # Find likelihood ratio
            likelihood_n = torch.exp(self.alpha * prod_2_n) / torch.exp(self.alpha * prod_1_n)

            # Select the better w
            rnd_n = torch.rand(likelihood_n.shape)
            choice_n = (rnd_n < likelihood_n) * 1

            ws_n2r = torch.stack([w1_nr, w2_nr], dim=1)

            w_nr = gather_by_actions(ws_n2r, choice_n)

            # Get the best actions for the selected w
            qa_tp1_values_na = (qa_tp1_values_nar * w_nr.unsqueeze(1).expand(qa_tp1_values_nar.shape)).sum(dim=2)

            ac_tp1_n = qa_tp1_values_na.argmax(dim=1)

            q_tp1_values_nr = gather_by_actions(qa_tp1_values_nar, ac_tp1_n)

        else:
            #####################
            # MDQN default
            #####################
            # Select the next action
            available_actions_n = [
                ParetoOptimalPolicy.find_strong_pareto_optimal_actions(vals, eps=self.eps)
                for vals in ptu.to_numpy(qa_tp1_values_nar)
            ]

            ac_tp1_n = np.array([np.random.choice(actions) for actions in available_actions_n])
            ac_tp1_n = ptu.from_numpy(ac_tp1_n).to(torch.long)

            # Find Q-values for the selected actions
            q_tp1_values_nr = gather_by_actions(qa_tp1_values_nar, ac_tp1_n)

            #####################
            # Consistent
            #####################
            if self.consistent:
                # Draw a new action (prime) and find its Q-values
                acp_tp1_n = np.array([np.random.choice(actions) for actions in available_actions_n])
                acp_tp1_n = ptu.from_numpy(acp_tp1_n).to(torch.long)
                qp_tp1_values_nr = gather_by_actions(qa_tp1_values_nar, acp_tp1_n)

                # Find inner-products of Q_a and Q_a'
                prod_t_tp1_n = (q_t_values_nr.detach() * q_tp1_values_nr).sum(dim=1)
                prodp_t_tp1_n = (q_t_values_nr.detach() * qp_tp1_values_nr).sum(dim=1)

                # Find likelihood ratio
                likelihood_n = torch.exp(self.alpha*prodp_t_tp1_n)/torch.exp((self.alpha*prod_t_tp1_n))

                # Select the better action
                rnd_n = torch.rand(likelihood_n.shape)
                choice_n = (rnd_n < likelihood_n)*1

                acs_tp1_n2 = torch.stack([ac_tp1_n, acp_tp1_n], dim=1)

                ac_tp1_n = gather_by_actions(acs_tp1_n2, choice_n)

                # Find Q-values for the selected actions
                q_tp1_values_nr = gather_by_actions(qa_tp1_values_nar, ac_tp1_n)

        # Compute targets for minimizing Bellman error
        # currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
        target_nr = reward_nr + self.gamma * q_tp1_values_nr * (1 - terminal_n.unsqueeze(1))
        target_nr = target_nr.detach()

        assert q_t_values_nr.shape == target_nr.shape

        loss = self.loss(q_t_values_nr, target_nr)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.mq_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        self.learning_rate_scheduler.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }

    def update_target_network(self):
        for target_param, param in zip(
                self.mq_net_target.parameters(), self.mq_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def qa_values(self, obs):
        obs = ptu.from_numpy(obs)
        qa_values = self.mq_net(obs)
        return ptu.to_numpy(qa_values)

    def save(self, save_path):
        torch.save(
            {
                'hparams': self.hparams,
                'optimizer_spec': self.optimizer_spec,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'q_net_state_dict': self.mq_net.state_dict(),
            }, save_path)

    @staticmethod
    def load(load_path):
        checkpoint = torch.load(load_path)

        dqn_critic = MDQNCritic(
            hparams=checkpoint['hparams'],
            optimizer_spec=checkpoint['optimizer_spec']
        )

        dqn_critic.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        dqn_critic.mq_net.load_state_dict(checkpoint['q_net_state_dict'])

        return dqn_critic


class ExtendedMDQNCritic(BaseCritic):

    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)

        self.hparams = hparams

        # Env
        self.env_name = hparams['env_name']
        self.ob_dim = hparams['ob_dim']

        if isinstance(self.ob_dim, int):
            self.input_shape = (self.ob_dim,)
        else:
            self.input_shape = hparams['input_shape']

        self.ac_dim = hparams['ac_dim']
        self.re_dim = hparams['re_dim']
        self.ex_dim = hparams['ex_dim']

        self.grad_norm_clipping = hparams['grad_norm_clipping']
        self.gamma = hparams['gamma']

        # Networks
        network_initializer = hparams['q_func']
        self.mq_net = network_initializer(self.ob_dim, self.ac_dim, self.re_dim, self.ex_dim)
        self.mq_net_target = network_initializer(self.ob_dim, self.ac_dim, self.re_dim, self.ex_dim)

        # Optimization
        self.optimizer_spec = optimizer_spec
        self.optimizer = self.optimizer_spec.constructor(
            self.mq_net.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )

        self.loss = nn.SmoothL1Loss()  # AKA Huber loss

        self.mq_net.to(ptu.device)
        self.mq_net_target.to(ptu.device)

        # Pruning
        self.consistent = hparams.get('consistent_emdqn', True)

        self.alpha = hparams['consistency_alpha']
        self.b = hparams['w_bound']

    def update(self, ob_no, ac_n, next_ob_no, reward_nr, terminal_n):
        """
            Update the parameters of the critic.
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_n = ptu.from_numpy(ac_n).to(torch.long)  # If action space is discrete, ac_n
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_nr = ptu.from_numpy(reward_nr)
        terminal_n = ptu.from_numpy(terminal_n)

        # Get the current Q-values
        qa_t_values_nare = self.mq_net(ob_no)
        q_t_values_nre = gather_by_actions(qa_t_values_nare, ac_n)

        # Compute the Q-values for the next observation
        qa_tp1_values_nare = self.mq_net_target(next_ob_no).detach()

        # Select the action for uniform consistent
        n, a, r, e = qa_tp1_values_nare.shape

        #####################
        # Consistent
        #####################
        if self.consistent:
            # Draw ws
            w1_n1r1 = torch.rand((n, 1, r, 1))*self.b + 1
            w2_n1r1 = torch.rand((n, 1, r, 1))*self.b + 1

            # Find the inner-products
            prod_1_na, _ = (qa_t_values_nare.detach() * w1_n1r1).sum(dim=2).max(dim=2)
            prod_2_na, _ = (qa_t_values_nare.detach() * w2_n1r1).sum(dim=2).max(dim=2)

            prod_1_n = gather_by_actions(prod_1_na, ac_n)
            prod_2_n = gather_by_actions(prod_2_na, ac_n)

            # Find the softmax probs
            prob_1_n = torch.exp(self.alpha * prod_1_n) / torch.exp(self.alpha * prod_1_na).sum(dim=1)
            prob_2_n = torch.exp(self.alpha * prod_2_n) / torch.exp(self.alpha * prod_2_na).sum(dim=1)

            # Find likelihood ratio
            likelihood_n = prob_2_n / prob_1_n

            # Select the better w
            rnd_n = torch.rand(likelihood_n.shape)
            choice_n = (rnd_n < likelihood_n) * 1

            w1_nr = w1_n1r1.squeeze()
            w2_nr = w2_n1r1.squeeze()
            ws_n2r = torch.stack([w1_nr, w2_nr], dim=1)

            w_nr = gather_by_actions(ws_n2r, choice_n)

            w_n1r1 = w_nr.unsqueeze(1).unsqueeze(3)

            # Get the best actions for the selected w
            qa_tp1_values_na, _ = (qa_tp1_values_nare * w_n1r1).sum(dim=2).max(dim=2)

            ac_tp1_n = qa_tp1_values_na.argmax(dim=1)

            q_tp1_values_nre = gather_by_actions(qa_tp1_values_nare, ac_tp1_n)

        #####################
        # Inconsistent
        #####################
        else:
            # Draw ws
            q_tp1_values_list = []
            for idx_e in range(e):
                w_n1r1 = torch.rand((n, 1, r, 1)) * self.b + 1

                # Get the best actions for w
                qa_w_tp1_values_na, _ = (qa_tp1_values_nare * w_n1r1).sum(dim=2).max(dim=2)

                ac_tp1_n = qa_w_tp1_values_na.argmax(dim=1)

                q_w_tp1_values_nre = gather_by_actions(qa_tp1_values_nare, ac_tp1_n)

                # Get the best ex dim for the best action
                w_nr1 = w_n1r1.squeeze(1)

                e_tp1_n = (q_w_tp1_values_nre * w_nr1).sum(dim=1).argmax(dim=1)

                q_w_tp1_values_nr = gather_by_e(q_w_tp1_values_nre, e_tp1_n)

                # Add to tle list
                q_tp1_values_list.append(q_w_tp1_values_nr)

            # Stack
            q_tp1_values_nre = torch.stack(q_tp1_values_list, dim=2)

        # Compute targets for minimizing Bellman error
        # currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
        target_nre = reward_nr.unsqueeze(2) + self.gamma * q_tp1_values_nre * (1 - terminal_n.unsqueeze(1).unsqueeze(2))
        target_nre = target_nre.detach()

        assert q_t_values_nre.shape == target_nre.shape

        loss = self.loss(q_t_values_nre, target_nre)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.mq_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        self.learning_rate_scheduler.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }

    def update_target_network(self):
        for target_param, param in zip(
                self.mq_net_target.parameters(), self.mq_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def qa_values(self, obs):
        obs = ptu.from_numpy(obs)
        qa_values = self.mq_net(obs)
        return ptu.to_numpy(qa_values)

    def save(self, save_path):
        torch.save(
            {
                'hparams': self.hparams,
                'optimizer_spec': self.optimizer_spec,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'q_net_state_dict': self.mq_net.state_dict(),
            }, save_path)

    @staticmethod
    def load(load_path):
        checkpoint = torch.load(load_path)

        dqn_critic = ExtendedMDQNCritic(
            hparams=checkpoint['hparams'],
            optimizer_spec=checkpoint['optimizer_spec']
        )

        dqn_critic.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        dqn_critic.mq_net.load_state_dict(checkpoint['q_net_state_dict'])

        return dqn_critic


class PrunedDQNCritic(DQNCritic):

    def __init__(self, hparams, optimizer_spec, action_pruner=None, **kwargs):
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
        choose_from_pruned = False if self.action_pruner is None else True

        if choose_from_pruned:
            available_actions = self.action_pruner.get_actions(ptu.to_numpy(ob_no))

        if self.double_q:
            # In double Q-learning, the best action is selected using the Q-network that
            # is being updated, but the Q-value for this action is obtained from the
            # target Q-network.

            qa_tp1_values = self.q_net(next_ob_no)

            if choose_from_pruned:
                ac_tp1 = get_maximizer_from_available_actions(qa_tp1_values, available_actions)
            else:
                ac_tp1 = qa_tp1_values.argmax(dim=1)

        else:
            if choose_from_pruned:
                ac_tp1 = get_maximizer_from_available_actions(qa_tp1_target_values, available_actions)
            else:
                ac_tp1 = qa_tp1_target_values.argmax(dim=1)

        q_tp1 = torch.gather(qa_tp1_target_values, 1, ac_tp1.unsqueeze(1)).squeeze(1)

        # Compute targets for minimizing Bellman error
        # currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
        target = reward_n + self.gamma * q_tp1 * (1 - terminal_n)
        target = target.detach()

        assert q_t_values.shape == target.shape
        loss = self.loss(q_t_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        self.learning_rate_scheduler.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }

    @staticmethod
    def load(load_path):
        checkpoint = torch.load(load_path)

        dqn_critic = PrunedDQNCritic(
            hparams=checkpoint['hparams'],
            optimizer_spec=checkpoint['optimizer_spec']
        )

        dqn_critic.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        dqn_critic.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        dqn_critic.q_net_target.load_state_dict(checkpoint['q_net_target_state_dict'])

        return dqn_critic
