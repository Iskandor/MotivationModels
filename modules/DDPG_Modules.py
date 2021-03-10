import copy

import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(Critic, self).__init__()

        self._hidden0 = nn.Linear(state_dim, config.critic_h1)
        self._hidden1 = nn.Linear(config.critic_h1 + action_dim, config.critic_h2)
        self._output = nn.Linear(config.critic_h2, 1)

        self.init()
        self.heads = 1

    def forward(self, features, action):
        x = torch.relu(self._hidden0(features))
        x = torch.cat([x, action], 1)
        x = torch.relu(self._hidden1(x))
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

    def forward(self, features):
        x = torch.relu(self._hidden0(features))
        x = torch.relu(self._hidden1(x))
        policy = torch.tanh(self._output(x))
        return policy

    def init(self):
        nn.init.xavier_uniform_(self._hidden0.weight)
        nn.init.xavier_uniform_(self._hidden1.weight)
        nn.init.uniform_(self._output.weight, -3e-1, 3e-1)


class DDPGNetwork(nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(DDPGNetwork, self).__init__()
        self.critic = Critic(input_shape, action_dim, config)
        self.actor = Actor(input_shape, action_dim, config)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
        self.hard_update()

    def value(self, state, action):
        value = self.critic(state, action)
        return value

    def action(self, state):
        policy = self.actor(state)
        return policy

    def value_target(self, state, action):
        value = self.critic_target(state, action)
        return value

    def action_target(self, state):
        policy = self.actor_target(state)
        return policy

    def soft_update(self, tau):
        self._soft_update(self.critic_target, self.critic, tau)
        self._soft_update(self.actor_target, self.actor, tau)

    def hard_update(self):
        self._hard_update(self.critic_target, self.critic)
        self._hard_update(self.actor_target, self.actor)

    @staticmethod
    def _soft_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    @staticmethod
    def _hard_update(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


class Critic2Heads(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(Critic2Heads, self).__init__()

        self._hidden0 = nn.Linear(state_dim, config.critic_h1)
        self._hidden1 = nn.Linear(config.critic_h1 + action_dim, config.critic_h2)
        self._output_ve = nn.Linear(config.critic_h2, 1)
        self._output_vi = nn.Linear(config.critic_h2, 1)

        self.init()
        self.heads = 2

    def forward(self, state, action):
        x = state
        x = torch.relu(self._hidden0(x))
        x = torch.cat([x, action], 1)
        x = torch.relu(self._hidden1(x))
        return self._output_ve(x), self._output_vi(x)

    def init(self):
        nn.init.xavier_uniform_(self._hidden0.weight)
        nn.init.xavier_uniform_(self._hidden1.weight)
        nn.init.uniform_(self._output_ve.weight, -3e-3, 3e-3)
        nn.init.uniform_(self._output_vi.weight, -3e-3, 3e-3)


class CriticDeep(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(CriticDeep, self).__init__()

        self._hidden0 = nn.Linear(state_dim, config.critic_h1)
        self._hidden1 = nn.Linear(config.critic_h1, config.critic_h1)
        self._hidden2 = nn.Linear(config.critic_h1 + action_dim, config.critic_h2)
        self._hidden3 = nn.Linear(config.critic_h2, config.critic_h2)
        self._output = nn.Linear(config.critic_h2, 1)

        self.init()
        self.heads = 1

    def forward(self, state, action):
        x = state
        x = torch.relu(self._hidden0(x))
        x = torch.relu(self._hidden1(x))
        x = torch.cat([x, action], 1)
        x = torch.relu(self._hidden2(x))
        x = torch.relu(self._hidden3(x))
        value = self._output(x)
        return value

    def init(self):
        nn.init.xavier_uniform_(self._hidden0.weight)
        nn.init.xavier_uniform_(self._hidden1.weight)
        nn.init.xavier_uniform_(self._hidden2.weight)
        nn.init.xavier_uniform_(self._hidden3.weight)
        nn.init.uniform_(self._output.weight, -3e-3, 3e-3)


class ActorDeep(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(ActorDeep, self).__init__()

        self._hidden0 = nn.Linear(state_dim, config.actor_h1)
        self._hidden1 = nn.Linear(config.actor_h1, config.actor_h1)
        self._hidden2 = nn.Linear(config.actor_h1, config.actor_h2)
        self._hidden3 = nn.Linear(config.actor_h2, config.actor_h2)
        self._output = nn.Linear(config.actor_h2, action_dim)

        self.init()

    def forward(self, state):
        x = state
        x = torch.relu(self._hidden0(x))
        x = torch.relu(self._hidden1(x))
        x = torch.relu(self._hidden2(x))
        x = torch.relu(self._hidden3(x))
        policy = torch.tanh(self._output(x))
        return policy

    def init(self):
        nn.init.xavier_uniform_(self._hidden0.weight)
        nn.init.xavier_uniform_(self._hidden1.weight)
        nn.init.xavier_uniform_(self._hidden2.weight)
        nn.init.xavier_uniform_(self._hidden3.weight)
        nn.init.uniform_(self._output.weight, -3e-1, 3e-1)
