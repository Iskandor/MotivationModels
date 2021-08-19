import torch

from algorithms.A2C import A2C
from algorithms.DDPG import DDPG


class M3Motivation:
    def __init__(self, gate, critic, gate_lr, critic_lr, gamma, tau, memory_buffer, sample_size, forward_model, meta_critic):
        self._gate = gate
        self._ddpg = DDPG(gate, critic, gate_lr, critic_lr, gamma, sample_size)
        self._forward_model = forward_model
        self._meta_critic = meta_critic

    def train(self, state0, action0, state1, action1, reward, done):
        error_estimate0 = self._meta_critic.error(state0, action0)
        error_estimate1 = self._meta_critic.error(state1, action1)
        m3_state0 = torch.cat([state0, error_estimate0.expand_as(state0)])
        m3_state1 = torch.cat([state1, error_estimate1.expand_as(state1)])
        m3_action = self._gate(m3_state0).detach()

        self._meta_critic.train(state0, action0, state1)
        self._ddpg.train(m3_state0, m3_action, m3_state1, reward, done)

    def reward(self, state0, action, state1):
        error = self._forward_model.error(state0, action, state1)
        error_estimate = self._meta_critic.error(state0, action)
        m3_state0 = torch.cat([state0, error_estimate.expand_as(state0)], dim=-1)
        gate = torch.softmax(self._gate(m3_state0).detach(), dim=-1)

        rewards = [self._curious_reward(error, error_estimate), self._familiar_reward(error, error_estimate), self._surprise_reward(error, error_estimate), self._predictive_penalty(error, error_estimate)]
        rewards = torch.stack(rewards).squeeze(state0.ndim)
        if state0.ndim == 2:
            rewards = rewards.T
        reward = torch.sum(rewards * gate, state0.ndim - 1).unsqueeze(state0.ndim - 1)

        return reward

    def get_forward_model(self):
        return self._forward_model

    def get_metacritic(self):
        return self._meta_critic

    @staticmethod
    def _curious_reward(error, error_estimate):
        return torch.tanh(error)

    @staticmethod
    def _familiar_reward(error, error_estimate):
        return torch.tanh(1 / error)

    @staticmethod
    def _surprise_reward(error, error_estimate):
        sigma = 1e-2
        mask = torch.gt(torch.abs(error - error_estimate), torch.ones_like(error) * sigma).type(torch.float32)
        return torch.tanh(error / error_estimate + error_estimate / error - 2) * mask

    @staticmethod
    def _predictive_penalty(error, error_estimate):
        return torch.tanh(error_estimate - error)
