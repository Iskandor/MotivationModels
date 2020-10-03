import torch

from algorithms.A2C import A2C


class M3Motivation:
    def __init__(self, gate, critic, gate_lr, critic_lr, gamma, forward_model, meta_critic):
        self._gate = gate
        self._a2c = A2C(gate, critic, gate_lr, critic_lr, gamma)
        self._forward_model = forward_model
        self._meta_critic = meta_critic

    def train(self, state0, action, state1, reward, done):
        self._meta_critic.train(state0, action, state1)
        self._a2c.train(state0, reward, done)

    def reward(self, state0, action, state1):
        error = self._forward_model.error(state0, action, state1)
        error_estimate = self._meta_critic.error(state0, action)
        gate = self._a2c.get_probs(state0)

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
