import numpy as np
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn

from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure.dqn_utils import get_maximizer_from_available_actions, gather_by_actions, gather_by_e
from cs285.critics.base_critic import BaseCritic
import cs285.policies.argmax_policy as argmax_policy
from cs285.policies.linearly_weighted_argmax_policy import draw_w
import cs285.policies.linearly_weighted_argmax_policy as lw_argmax_policy
from cs285.pruners.base_pruner import BasePruner


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
        self.re_dim = hparams.get('re_dim', 1)
        self.ex_dim = hparams.get('ex_dim', 1)

        self.double_q = hparams['double_q']
        self.grad_norm_clipping = hparams['grad_norm_clipping']
        self.gamma = hparams['gamma']

        # Networks
        network_initializer = hparams['q_func']
        self.q_net = network_initializer(self.ob_dim, self.ac_dim, self.re_dim, self.ex_dim)
        self.q_net_target = network_initializer(self.ob_dim, self.ac_dim, self.re_dim, self.ex_dim)

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

        # CQL
        self.add_cql_loss = hparams.get('add_cql_loss', False)
        if self.add_cql_loss:
            self.cql_alpha = hparams['cql_alpha']

    def get_actor_class(self):
        return argmax_policy.ArgMaxPolicy

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

        # Add CQL loss if requested
        if self.add_cql_loss:
            q_t_logsumexp = torch.logsumexp(qa_t_values, dim=1)
            cql_loss = torch.mean(q_t_logsumexp - q_t_values)
            loss = self.cql_alpha * cql_loss + loss

        # Step the optimizer
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

    @classmethod
    def load(cls, load_path):
        checkpoint = torch.load(load_path)

        dqn_critic = cls(
            hparams=checkpoint['hparams'],
            optimizer_spec=checkpoint['optimizer_spec']
        )

        dqn_critic.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        dqn_critic.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        dqn_critic.q_net_target.load_state_dict(checkpoint['q_net_target_state_dict'])

        dqn_critic.q_net.to(ptu.device)
        dqn_critic.q_net_target.to(ptu.device)

        return dqn_critic


class PrunedDQNCritic(DQNCritic):

    def __init__(self, hparams, optimizer_spec, action_pruner: BasePruner, **kwargs):
        super().__init__(hparams, optimizer_spec, **kwargs)

        # Pruning
        self.action_pruner = action_pruner

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
            Update the parameters of the critic.
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
        available_actions = self.action_pruner.get_list_of_available_actions(ptu.to_numpy(ob_no))

        if self.double_q:
            # In double Q-learning, the best action is selected using the Q-network that
            # is being updated, but the Q-value for this action is obtained from the
            # target Q-network.

            qa_tp1_values = self.q_net(next_ob_no)

            ac_tp1 = get_maximizer_from_available_actions(qa_tp1_values, available_actions)
        else:
            ac_tp1 = get_maximizer_from_available_actions(qa_tp1_target_values, available_actions)

        q_tp1 = torch.gather(qa_tp1_target_values, 1, ac_tp1.unsqueeze(1)).squeeze(1)

        # Compute targets for minimizing Bellman error
        # currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
        target = reward_n + self.gamma * q_tp1 * (1 - terminal_n)
        target = target.detach()

        assert q_t_values.shape == target.shape
        loss = self.loss(q_t_values, target)

        # Add CQL loss if requested
        if self.add_cql_loss:
            q_t_logsumexp = torch.logsumexp(qa_t_values, dim=1)
            cql_loss = torch.mean(q_t_logsumexp - q_t_values)
            loss = self.cql_alpha * cql_loss + loss

        # Step the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        self.learning_rate_scheduler.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }

    def get_actor_class(self):
        return argmax_policy.PrunedArgMaxPolicy


class MDQNCritic(DQNCritic):

    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(hparams, optimizer_spec, **kwargs)

        self.optimistic = hparams.get('optimistic_mdqn', False)
        self.diverse = hparams.get('diverse_mdqn', False)
        self.consistent = hparams.get('consistent_mdqn', False)
        if np.sum([self.optimistic, self.diverse, self.consistent]) != 1:
            raise Exception('MDQN type is inconsistently defined!')

        self.b = hparams['w_bound']
        if self.consistent:
            self.alpha = hparams['consistency_alpha']

    def get_actor_class(self):
        if self.optimistic:
            return lw_argmax_policy.LinearlyWeightedArgMaxPolicy
        elif self.diverse:
            return lw_argmax_policy.LinearlyWeightedArgMaxPolicy
        elif self.consistent:
            return lw_argmax_policy.LinearlyWeightedSoftmaxPolicy
        else:
            raise Exception('Invalid MDQN type!')

    def update(self, ob_no, ac_n, next_ob_no, reward_nr, terminal_n):
        """
            Update the parameters of the critic.
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_n = ptu.from_numpy(ac_n).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_nr = ptu.from_numpy(reward_nr)
        terminal_n = ptu.from_numpy(terminal_n)

        # Get the current Q-values
        qa_t_values_nar = self.q_net(ob_no)
        q_t_values_nr = gather_by_actions(qa_t_values_nar, ac_n)

        # Compute the Q-values for the next observation
        qa_tp1_values_nar = self.q_net_target(next_ob_no)

        # Find next actions and Q-values based on MDQN type
        actor_class = self.get_actor_class()

        #####################
        # Optimistic
        #####################
        if self.optimistic:
            if self.double_q:
                ac_tp1_nr = self.q_net(next_ob_no).argmax(dim=1)
                q_tp1_values_nr = torch.gather(qa_tp1_values_nar, 1, ac_tp1_nr.unsqueeze(1)).squeeze(1)
            else:
                q_tp1_values_nr, _ = qa_tp1_values_nar.max(dim=1)

        #####################
        # Consistent
        #####################
        elif self.consistent:
            n, a, r = qa_t_values_nar.shape

            # Draw ws
            w1_n1r = ptu.from_numpy(draw_w((n, 1, r), self.b))
            w2_n1r = ptu.from_numpy(draw_w((n, 1, r), self.b))

            # Find the inner-products
            prod_1_na = (qa_t_values_nar * w1_n1r).sum(dim=2)
            prod_2_na = (qa_t_values_nar * w2_n1r).sum(dim=2)

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

            ws_n2r = torch.stack([w1_n1r.squeeze(1), w2_n1r.squeeze(1)], dim=1)

            w_nr = gather_by_actions(ws_n2r, choice_n)

            # Get the best actions for the selected w
            if self.double_q:
                ac_tp1_n = actor_class.get_actions_for_w(self.q_net(next_ob_no), w_nr, alpha=self.alpha)
            else:
                ac_tp1_n = actor_class.get_actions_for_w(qa_tp1_values_nar, w_nr, alpha=self.alpha)

            q_tp1_values_nr = gather_by_actions(qa_tp1_values_nar, ac_tp1_n)

        #####################
        # Diverse
        #####################
        else:
            n, a, r = qa_t_values_nar.shape

            # Draw w
            w_nr = ptu.from_numpy(draw_w((n, r), self.b))

            # Get the best actions for the selected w
            if self.double_q:
                ac_tp1_n = actor_class.get_actions_for_w(self.q_net(next_ob_no), w_nr)
            else:
                ac_tp1_n = actor_class.get_actions_for_w(qa_tp1_values_nar, w_nr)

            q_tp1_values_nr = gather_by_actions(qa_tp1_values_nar, ac_tp1_n)

        # Compute targets for minimizing Bellman error
        # currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
        target_nr = reward_nr + self.gamma * q_tp1_values_nr * (1 - terminal_n.unsqueeze(1))
        target_nr = target_nr.detach()

        assert q_t_values_nr.shape == target_nr.shape

        loss = self.loss(q_t_values_nr, target_nr)

        # Add CQL loss if requested
        if self.add_cql_loss:
            q_t_logsumexp_nr = torch.logsumexp(qa_t_values_nar, dim=1)
            cql_loss = torch.mean(q_t_logsumexp_nr - q_t_values_nr)
            loss = self.cql_alpha * cql_loss + loss

        # Step the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        self.learning_rate_scheduler.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }


class ExtendedMDQNCritic(DQNCritic):

    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(hparams, optimizer_spec, **kwargs)

        # Pruning
        self.diverse = hparams.get('diverse_emdqn', False)
        self.consistent = hparams.get('consistent_emdqn', False)
        if np.sum([self.diverse, self.consistent]) != 1:
            raise Exception('EMDQN type is inconsistently defined!')

        self.b = hparams['w_bound']
        if self.consistent:
            self.alpha = hparams['consistency_alpha']

    def get_actor_class(self):
        if self.diverse:
            return lw_argmax_policy.ExtendedLinearlyWeightedArgMaxPolicy
        elif self.consistent:
            return lw_argmax_policy.ExtendedLinearlyWeightedSoftmaxPolicy
        else:
            raise Exception('Invalid EMDQN type!')

    def update(self, ob_no, ac_n, next_ob_no, reward_nr, terminal_n):
        """
            Update the parameters of the critic.
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_n = ptu.from_numpy(ac_n).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_nr = ptu.from_numpy(reward_nr)
        terminal_n = ptu.from_numpy(terminal_n)

        # Get the current Q-values
        qa_t_values_nare = self.q_net(ob_no)
        q_t_values_nre = gather_by_actions(qa_t_values_nare, ac_n)

        # Compute the Q-values for the next observation
        qa_tp1_values_nare = self.q_net_target(next_ob_no).detach()

        # Find next actions and Q-values based on MDQN type
        actor_class = self.get_actor_class()

        n, a, r, e = qa_tp1_values_nare.shape

        #####################
        # Consistent
        #####################
        if self.consistent:
            # Draw ws
            w1_n1r1 = ptu.from_numpy(draw_w((n, 1, r, 1), self.b))
            w2_n1r1 = ptu.from_numpy(draw_w((n, 1, r, 1), self.b))

            # Find the inner-products
            prod_1_na, _ = (qa_t_values_nare * w1_n1r1).sum(dim=2).max(dim=2)
            prod_2_na, _ = (qa_t_values_nare * w2_n1r1).sum(dim=2).max(dim=2)

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

            # Get the best actions for the selected w
            if self.double_q:
                ac_tp1_n = actor_class.get_actions_for_w(self.q_net(next_ob_no), w_nr, alpha=self.alpha)
            else:
                ac_tp1_n = actor_class.get_actions_for_w(qa_tp1_values_nare, w_nr, alpha=self.alpha)

            q_tp1_values_nre = gather_by_actions(qa_tp1_values_nare, ac_tp1_n)

        #####################
        # Diverse
        #####################
        else:
            q_tp1_values_list = []
            for idx_e in range(e):
                # Draw ws
                w_nr = ptu.from_numpy(draw_w((n, r), self.b))

                # Get the best actions for w
                if self.double_q:
                    ac_tp1_n = actor_class.get_actions_for_w(self.q_net(next_ob_no), w_nr)
                else:
                    ac_tp1_n = actor_class.get_actions_for_w(qa_tp1_values_nare, w_nr)

                q_w_tp1_values_nre = gather_by_actions(qa_tp1_values_nare, ac_tp1_n)

                # Get the best ex dim for the best action
                w_nr1 = w_nr.unsqueeze(2)

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

        # Add CQL loss if requested
        if self.add_cql_loss:
            q_t_logsumexp_nre = torch.logsumexp(qa_t_values_nare, dim=1)
            cql_loss = torch.mean(q_t_logsumexp_nre - q_t_values_nre)
            loss = self.cql_alpha * cql_loss + loss

        # Step the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        self.learning_rate_scheduler.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }
