import numpy as np
import torch
import torch.nn as nn

from agents import TYPE
from modules import init_orthogonal, init_xavier_uniform
from modules.PPO_Modules import ContinuousHead, Actor, Critic2Heads, ActorNHeads, DiscreteHead
from modules.rnd_models.RNDModelAeris import RNDModelAeris, QRNDModelAeris, DOPModelAeris


class PPOAerisNetwork(torch.nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(PPOAerisNetwork, self).__init__()

        self.channels = input_shape[0]
        self.width = input_shape[1]

        self.features = nn.Sequential(
            nn.Conv1d(self.channels, config.critic_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        init_orthogonal(self.features[0], np.sqrt(2))

        fc_count = config.critic_kernels_count * self.width // 4

        self.critic = nn.Sequential(
            # nn.Conv1d(self.channels, config.critic_kernels_count, kernel_size=8, stride=4, padding=2),
            # nn.ReLU(),
            # nn.Flatten(),
            nn.Linear(fc_count, config.critic_h1),
            nn.ReLU(),
            nn.Linear(config.critic_h1, 1))

        init_xavier_uniform(self.critic[0])
        init_xavier_uniform(self.critic[2])
        # nn.init.uniform_(self.critic[5].weight, -0.003, 0.003)

        fc_count = config.actor_kernels_count * self.width // 4

        self.layers_actor = nn.Sequential(
            # nn.Conv1d(self.channels, config.actor_kernels_count, kernel_size=8, stride=4, padding=2),
            # nn.ReLU(),
            # nn.Flatten(),
            nn.Linear(fc_count, config.actor_h1),
            nn.ReLU(),
            ContinuousHead(config.actor_h1, action_dim))

        init_xavier_uniform(self.layers_actor[0])
        # nn.init.xavier_uniform_(self.layers_actor[3].weight)

        self.actor = Actor(self.layers_actor, TYPE.continuous, action_dim)

    def forward(self, state):
        x = self.features(state)
        # x = state
        value = self.critic(x)
        action, probs = self.actor(x)

        return value, torch.tanh(action), probs


class PPOAerisMotivationNetwork(PPOAerisNetwork):
    def __init__(self, input_shape, action_dim, config):
        super(PPOAerisMotivationNetwork, self).__init__(input_shape, action_dim, config)

        self.channels = input_shape[0]
        self.width = input_shape[1]

        fc_count = config.critic_kernels_count * self.width // 4

        self.critic = nn.Sequential(
            nn.Linear(fc_count, config.critic_h1),
            nn.ReLU(),
            Critic2Heads(config.critic_h1))

        nn.init.xavier_uniform_(self.critic[0].weight)
        nn.init.zeros_(self.critic[0].bias)


class PPOAerisNetworkRND(PPOAerisMotivationNetwork):
    def __init__(self, input_shape, action_dim, config):
        super(PPOAerisNetworkRND, self).__init__(input_shape, action_dim, config)
        self.rnd_model = RNDModelAeris(input_shape, action_dim, config)


class PPOAerisNetworkDOP(PPOAerisNetwork):
    def __init__(self, input_shape, action_dim, config):
        super(PPOAerisNetworkDOP, self).__init__(input_shape, action_dim, config)

        self.action_dim = action_dim
        self.channels = input_shape[0]
        self.width = input_shape[1]
        self.head_count = 4

        fc_count = config.critic_kernels_count * self.width // 4

        self.layers_actor = nn.Sequential(
            nn.Linear(fc_count, fc_count),
            nn.ReLU(),
            ActorNHeads(TYPE.continuous, self.head_count, [fc_count, config.actor_h1, action_dim], config)
        )

        nn.init.xavier_uniform_(self.layers_actor[0].weight)
        nn.init.zeros_(self.layers_actor[0].bias)

        self.actor = ActorNHeads(self.head_count, action_dim, self.layers_actor, TYPE.continuous, config)
        self.motivator = QRNDModelAeris(input_shape, action_dim * 2, config)
        self.dop_model = DOPModelAeris(self.head_count, input_shape, action_dim * 2, config, self.features, self.actor, self.motivator)
        self.indices = []

    def forward(self, state):
        x = self.features(state)
        value = self.critic(x)
        action, probs = self.actor(x)

        state = state.unsqueeze(1).repeat(1, self.head_count, 1, 1).view(-1, self.channels, self.width)
        probs = probs.view(-1, self.action_dim * 2)

        error = self.motivator.error(state, probs).view(-1, self.head_count).detach()
        argmax = error.argmax(dim=1)
        probs = probs.view(-1, self.head_count, self.action_dim * 2)
        action = action.gather(dim=1, index=argmax.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.action_dim))
        probs = probs.gather(dim=1, index=argmax.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.action_dim * 2))

        return value, (action.squeeze(1), argmax), probs.squeeze(1)


class PPOAerisNetworkDOPRef(PPOAerisNetwork):
    def __init__(self, input_shape, action_dim, config):
        super(PPOAerisNetworkDOPRef, self).__init__(input_shape, action_dim, config)

        self.config = config
        self.action_dim = action_dim
        self.channels = input_shape[0]
        self.width = input_shape[1]
        self.head_count = 4

        fc_count = config.critic_kernels_count * self.width // 4

        self.layers_actor = nn.Sequential(
            nn.Linear(fc_count, fc_count),
            nn.ReLU(),
            ActorNHeads(TYPE.continuous, self.head_count, [fc_count, config.actor_h1, action_dim], config)
        )

        nn.init.xavier_uniform_(self.layers_actor[0].weight)
        nn.init.zeros_(self.layers_actor[0].bias)

        self.actor = Actor(self.layers_actor, TYPE.continuous, action_dim)
        self.indices = []

    def forward(self, state):
        x = self.features(state)
        value = self.critic(x)
        action, probs = self.actor(x)

        argmax = torch.randint(0, self.head_count, (state.shape[0],), device=self.config.device)
        action = action.gather(dim=1, index=argmax.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.action_dim))
        probs = probs.gather(dim=1, index=argmax.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.action_dim * 2))

        return value, action.squeeze(1), probs.squeeze(1)


class PPOAerisGridNetwork(torch.nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(PPOAerisGridNetwork, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_shape[0], 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            # nn.AvgPool1d(input_shape[1]),
            nn.Flatten(),
            nn.Linear(input_shape[1] * 128, 256),
            nn.ReLU(),
        )

        init_orthogonal(self.features[0], np.sqrt(2))
        init_orthogonal(self.features[3], np.sqrt(2))

        self.critic = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1))

        init_orthogonal(self.critic[0], np.sqrt(2))
        init_orthogonal(self.critic[2], 0.01)

        self.layers_actor = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            DiscreteHead(256, action_dim))

        init_orthogonal(self.layers_actor[0], 0.01)
        init_orthogonal(self.layers_actor[2], 0.01)

        self.actor = Actor(self.layers_actor, TYPE.discrete, action_dim)

    def forward(self, state):
        x = self.features(state)
        value = self.critic(x)
        action, probs = self.actor(x)
        action = self.actor.encode_action(action)

        return value, action, probs
