import torch

from algorithms.DDPG import DDPG


class M2Motivation:
    def __init__(self, gate, gate_lr, gamma, tau, memory_buffer, sample_size, actor, critic, forward_model):
        self._gate = gate
        self._gate_agent = DDPG(gate.actor, gate.critic, gate_lr, gate_lr * 2, gamma, tau, memory_buffer, sample_size)
        self._critic = critic
        self._actor = actor
        self._forward_model = forward_model

    def train(self, state0, state1, reward, done):
        a0 = self._actor(state0).detach()
        v0 = self._critic(state0, a0).detach()
        a1 = self._actor(state1).detach()
        v1 = self._critic(state1, a1).detach()
        m3_state0 = torch.cat([state0, v0.expand_as(state0)], dim=1)
        m3_state1 = torch.cat([state1, v1.expand_as(state1)], dim=1)
        m3_action = torch.softmax(self._gate.policy(m3_state0).detach(), dim=1)

        self._gate_agent.train(m3_state0, m3_action, m3_state1, reward, done)

    def reward(self, state0, action, state1):
        error = self._forward_model.error(state0, action, state1)
        rewards = torch.cat([self._curious_reward(error), self._familiar_reward(error)], dim=1)

        weight = self.weight(state0, action)
        reward = rewards * weight
        reward = reward.sum(dim=1).unsqueeze(1)

        return reward

    def weight(self, state0, action):
        v0 = self._critic(state0, action).detach()
        m3_state0 = torch.cat([state0, v0.expand_as(state0)], dim=1)
        weight = torch.softmax(self._gate.policy(m3_state0).detach(), dim=1)

        return weight

    def get_forward_model(self):
        return self._forward_model

    @staticmethod
    def _curious_reward(error):
        return torch.tanh(error)

    @staticmethod
    def _familiar_reward(error):
        return torch.tanh(1 / error)

    def save(self, path):
        self._forward_model.save(path)
        torch.save(self._gate.state_dict(), path + '_m2.pth')

    def load(self, path):
        self._forward_model.load(path)
        self._gate.load_state_dict(torch.load(path + '_m2.pth'))
