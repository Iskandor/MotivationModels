import copy

import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np

from modules import init_xavier_uniform, init_custom

from modules.forward_models.ForwardModelBullet import ForwardModelBullet
from modules.metacritic_models.MetaCriticModelBullet import MetaCriticModelBullet, MetaCriticRNDModelBullet
from modules.rnd_models.RNDModelBullet import RNDModelBullet, QRNDModelBullet, DOPModelBullet, DOPSimpleModelBullet


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


class ActorNHeads(nn.Module):
    def __init__(self, head_count, action_dim, layers, config):
        super(ActorNHeads, self).__init__()

        self.actor = nn.Sequential(*layers)
        self.head_count = head_count
        self.heads = nn.ModuleList([nn.Sequential(
            nn.Linear(config.actor_h1, config.actor_h1),
            nn.ReLU(),
            nn.Linear(config.actor_h1, action_dim),
            nn.Tanh())
            for _ in range(head_count)])

        # for h in self.heads:
        #     init_xavier_uniform(h[0])
        #     init_xavier_uniform(h[2])

        weight1 = torch.zeros(head_count * config.actor_h1, config.actor_h1)
        nn.init.orthogonal_(weight1, 3)
        weight1 = weight1.reshape(head_count, config.actor_h1, config.actor_h1)
        weight2 = torch.zeros(head_count * action_dim, config.actor_h1)
        nn.init.orthogonal_(weight2, 3)
        weight2 = weight2.reshape(head_count, action_dim, config.actor_h1)

        for i, h in enumerate(self.heads):
            init_custom(h[0], weight1[i])
            init_custom(h[2], weight2[i])

    def forward(self, x):
        x = self.actor(x)
        actions = []

        for h in self.heads:
            a = h(x)
            actions.append(a)

        return torch.stack(actions, dim=1)


class DDPGNetwork(nn.Module):
    def __init__(self):
        super(DDPGNetwork, self).__init__()
        self.critic_heads = 1
        self.critic = None
        self.actor = None
        self.critic_target = None
        self.actor_target = None

    def value(self, state, action):
        x = torch.cat([state, action], dim=1)
        value = self.critic(x)
        return value

    def action(self, state):
        policy = self.actor(state)
        return policy

    def value_target(self, state, action):
        with torch.no_grad():
            x = torch.cat([state, action], dim=1)
            value = self.critic_target(x)
        return value

    def action_target(self, state):
        with torch.no_grad():
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


class DDPGSimpleNetwork(DDPGNetwork):
    def __init__(self, input_shape, action_dim, config):
        super(DDPGSimpleNetwork, self).__init__()
        self.critic = Critic(input_shape, action_dim, config)
        self.actor = Actor(input_shape, action_dim, config)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
        self.hard_update()


class DDPGBulletNetwork(DDPGNetwork):
    def __init__(self, state_dim, action_dim, config):
        super(DDPGBulletNetwork, self).__init__()

        critic_h = [int(x) for x in config.critic_h.split(',')]

        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, critic_h[0]),
            nn.ReLU(),
            nn.Linear(critic_h[0], critic_h[1]),
            nn.ReLU(),
            nn.Linear(critic_h[1], 1))

        nn.init.xavier_uniform_(self.critic[0].weight)
        nn.init.xavier_uniform_(self.critic[2].weight)
        nn.init.uniform_(self.critic[4].weight, -0.003, 0.003)

        actor_h = [int(x) for x in config.actor_h.split(',')]

        self.actor = nn.Sequential(
            nn.Linear(state_dim, actor_h[0]),
            nn.ReLU(),
            nn.Linear(actor_h[0], actor_h[1]),
            nn.ReLU(),
            nn.Linear(actor_h[1], action_dim),
            nn.Tanh())

        nn.init.xavier_uniform_(self.actor[0].weight)
        nn.init.xavier_uniform_(self.actor[2].weight)
        nn.init.uniform_(self.actor[4].weight, -0.3, 0.3)

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
        self.hard_update()


class DDPGBulletNetworkFM(DDPGBulletNetwork):
    def __init__(self, state_dim, action_dim, config):
        super(DDPGBulletNetworkFM, self).__init__(state_dim, action_dim, config)
        self.forward_model = ForwardModelBullet(state_dim, action_dim, config)


class DDPGBulletNetworkSU(DDPGBulletNetworkFM):
    def __init__(self, state_dim, action_dim, config):
        super(DDPGBulletNetworkSU, self).__init__(state_dim, action_dim, config)
        self.metacritic_model = MetaCriticModelBullet(self.forward_model, state_dim, action_dim, config)


class DDPGBulletNetworkRND(DDPGBulletNetwork):
    def __init__(self, state_dim, action_dim, config):
        super(DDPGBulletNetworkRND, self).__init__(state_dim, action_dim, config)
        self.rnd_model = RNDModelBullet(state_dim, action_dim, config)


class DDPGBulletNetworkQRND(DDPGBulletNetwork):
    def __init__(self, state_dim, action_dim, config):
        super(DDPGBulletNetworkQRND, self).__init__(state_dim, action_dim, config)
        self.qrnd_model = QRNDModelBullet(state_dim, action_dim, config)


class DDPGBulletNetworkDOPSimple(DDPGBulletNetwork):
    def __init__(self, state_dim, action_dim, config):
        super(DDPGBulletNetworkDOPSimple, self).__init__(state_dim, action_dim, config)
        self.dop_model = DOPSimpleModelBullet(state_dim, action_dim, config, self.actor)


class DDPGBulletNetworkDOP(DDPGBulletNetwork):
    def __init__(self, state_dim, action_dim, config):
        super(DDPGBulletNetworkDOP, self).__init__(state_dim, action_dim, config)
        self.dop_model = DOPModelBullet(state_dim, action_dim, config)

    def noise(self, state, action):
        return self.dop_model.noise(state, action)


class DDPGBulletNetworkSURND(DDPGBulletNetwork):
    def __init__(self, state_dim, action_dim, config):
        super(DDPGBulletNetworkSURND, self).__init__(state_dim, action_dim, config)
        self.rnd_model = RNDModelBullet(state_dim, action_dim, config)
        self.metacritic_model = MetaCriticRNDModelBullet(self.rnd_model, state_dim, action_dim, config)
