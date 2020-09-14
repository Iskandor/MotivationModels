import abc

import torch


class M3Critic(torch.nn.Module):
    @abc.abstractmethod
    def __init__(self, state_dim):
        super(M3Critic, self).__init__()

    @abc.abstractmethod
    def forward(self, state, error, error_estimate):
        raise NotImplementedError


class M3Gate(torch.nn.Module):
    @abc.abstractmethod
    def __init__(self, state_dim):
        super(M3Gate, self).__init__()

    @abc.abstractmethod
    def forward(self, state, error, error_estimate):
        raise NotImplementedError

class M3Motivation:
    def __init__(self, gate, critic, gate_lr, critic_lr, gamma, forward_model, meta_critic):
        self._gate = gate
        self._critic = critic
        self._forward_model = forward_model
        self._meta_critic = meta_critic

        self._gamma = gamma

        self._gate_optimizer = torch.optim.Adam(self._gate.parameters(), lr=gate_lr)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=critic_lr)

    def train(self, state0, action0, reward, state1, action1, done):
        error_estimate_t0 = self._meta_critic.error(state0, action0)
        value_estimate_t0 = self._critic(state0, error_estimate_t0.expand_as(state0))
        error_estimate_t1 = self._meta_critic.error(state1, action1)
        value_estimate_t1 = self._critic(state1, error_estimate_t1.expand_as(state1))
        expected_value = reward + (1 - done) * (self._gamma * value_estimate_t1)

        self._critic_optimizer.zero_grad()
        value_loss = torch.nn.functional.mse_loss(value_estimate_t0, expected_value)
        value_loss.backward()
        self._critic_optimizer.step()

    def reward(self, state0, action, state1):
        error = self._forward_model.error(state0, action, state1)
        error_estimate = self._meta_critic.error(state0, action)

