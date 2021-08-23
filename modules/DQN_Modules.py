import copy

import torch
from torch import nn

from modules import init_xavier_uniform, init_uniform


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(Critic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(state_dim, config.critic_h1),
            nn.ReLU(),
            nn.Linear(config.critic_h1, config.critic_h2),
            nn.ReLU(),
            nn.Linear(config.critic_h2, action_dim)
        )

        init_xavier_uniform(self.critic[0])
        init_xavier_uniform(self.critic[2])
        init_uniform(self.critic[4], 3e-3)

    def forward(self, state):
        return self.critic(state)


class DQNNetwork(nn.Module):
    def __init__(self):
        super(DQNNetwork, self).__init__()
        self.critic = None
        self.critic_target = None

    def value(self, state):
        value = self.critic(state)
        return value

    def value_target(self, state):
        with torch.no_grad():
            value = self.critic_target(state)
        return value

    def hard_update(self):
        self._hard_update(self.critic_target, self.critic)

    @staticmethod
    def _hard_update(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


class DQNSimpleNetwork(DQNNetwork):
    def __init__(self, input_shape, action_dim, config):
        super(DQNSimpleNetwork, self).__init__()
        self.critic = Critic(input_shape, action_dim, config)
        self.critic_target = copy.deepcopy(self.critic)
        self.hard_update()
