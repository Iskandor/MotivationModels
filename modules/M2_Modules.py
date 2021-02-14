import torch
from torch import nn


class M2Gate(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(M2Gate, self).__init__()

        self.critic = Critic(state_dim, action_dim, config)
        self.actor = Actor(state_dim, action_dim, config)

    def value(self, x):
        return self.critic(x)

    def policy(self, x):
        return self.actor(x)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(Critic, self).__init__()

        self._hidden0 = nn.Linear(state_dim, config.critic_h1)
        self._hidden1 = nn.Linear(config.critic_h1 + action_dim, config.critic_h2)
        self._output = nn.Linear(config.critic_h2, 1)

        self.init()

    def forward(self, state, action):
        x = state
        x = torch.tanh(self._hidden0(x))
        x = torch.cat([x, action], 1)
        x = torch.tanh(self._hidden1(x))
        value = self._output(x)
        return value

    def init(self):
        nn.init.xavier_uniform_(self._hidden0.weight)
        nn.init.xavier_uniform_(self._hidden1.weight)
        nn.init.uniform_(self._output.weight, -3e-3, 3e-3)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(Actor, self).__init__()

        self._hidden0 = nn.Linear(state_dim, config.actor_h1)
        self._hidden1 = nn.Linear(config.actor_h1, config.actor_h2)
        self._output = nn.Linear(config.actor_h2, action_dim)

        self.init()

    def forward(self, state):
        x = state
        x = torch.tanh(self._hidden0(x))
        x = torch.tanh(self._hidden1(x))
        policy = self._output(x)
        return policy

    def init(self):
        nn.init.xavier_uniform_(self._hidden0.weight)
        nn.init.xavier_uniform_(self._hidden1.weight)
        nn.init.uniform_(self._output.weight, -3e-1, 3e-1)
