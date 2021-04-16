import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical, MultivariateNormal, Normal

from agents import TYPE


class DiscreteHead(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DiscreteHead, self).__init__()
        self.logits = nn.Linear(input_dim, action_dim, bias=True)

        torch.nn.init.xavier_uniform_(self.logits.weight)

    def forward(self, x):
        logits = self.logits(x)
        probs = torch.softmax(logits, dim=1)
        dist = Categorical(probs)

        action = dist.sample().unsqueeze(1)

        return action, probs


class ContinuousHead(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ContinuousHead, self).__init__()
        self.mu = nn.Sequential(
            nn.Linear(input_dim, action_dim),
            nn.Tanh()
        )
        self.var = nn.Sequential(
            nn.Linear(input_dim, action_dim),
            nn.Softplus()
        )

        torch.nn.init.xavier_uniform_(self.mu[0].weight)
        torch.nn.init.xavier_uniform_(self.var[0].weight)

        self.action_dim = action_dim

    def forward(self, x):
        mu = self.mu(x)
        var = self.var(x)

        dist = Normal(mu, var.sqrt())
        action = dist.sample()

        return action, torch.cat([mu, var], dim=1)


class PPOSimpleNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, config, head):
        super(PPOSimpleNetwork, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(state_dim, config.critic_h1),
            nn.ReLU(),
            nn.Linear(config.critic_h1, config.critic_h2),
            nn.ReLU(),
            torch.nn.Linear(config.critic_h2, 1)
        )
        nn.init.xavier_uniform_(self.critic[0].weight)
        nn.init.xavier_uniform_(self.critic[2].weight)

        self.layers_actor = [
            torch.nn.Linear(state_dim, config.actor_h1),
            nn.ReLU(),
            torch.nn.Linear(config.actor_h1, config.actor_h2),
            nn.ReLU()
        ]
        nn.init.xavier_uniform_(self.layers_actor[0].weight)
        nn.init.xavier_uniform_(self.layers_actor[2].weight)

        if head == TYPE.discrete:
            self.layers_actor.append(DiscreteHead(config.actor_h2, action_dim))
        if head == TYPE.continuous:
            self.layers_actor.append(ContinuousHead(config.actor_h2, action_dim))
        if head == TYPE.multibinary:
            pass

        self.actor = nn.Sequential(*self.layers_actor)

    def forward(self, state):
        value = self.critic(state)
        action, probs = self.actor(state)

        return value, action, probs


class PPOAerisNetwork(torch.nn.Module):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAerisNetwork, self).__init__()

        self.channels = input_shape[0]
        self.width = input_shape[1]

        fc_count = config.critic_kernels_count * self.width // 4

        self.critic = nn.Sequential(
            nn.Conv1d(self.channels, config.critic_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_count, config.critic_h1),
            nn.ReLU(),
            nn.Linear(config.critic_h1, 1))

        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.xavier_uniform_(self.layers[3].weight)
        nn.init.uniform_(self.layers[5].weight, -0.003, 0.003)

        self.channels = input_shape[0]
        self.width = input_shape[1]

        fc_count = config.actor_kernels_count * self.width // 4

        self.layers_actor = [
            nn.Conv1d(self.channels, config.actor_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_count, config.actor_h1),
            nn.ReLU()]

        if head == TYPE.discrete:
            self.layers_actor.append(DiscreteHead(config.actor_h2, action_dim))
        if head == TYPE.continuous:
            self.layers_actor.append(ContinuousHead(config.actor_h2, action_dim))
        if head == TYPE.multibinary:
            pass

        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.xavier_uniform_(self.layers[3].weight)

        self.actor = nn.Sequential(*self.layers_actor)

    def forward(self, state):
        value = self.critic(state)
        action, probs = self.actor(state)

        return value, action, probs


class PPOAtariNetwork(torch.nn.Module):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAtariNetwork, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim
        input_channels = self.input_shape[0]
        self.feature_dim = 448

        self.layers_features = [
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, self.feature_dim),
            nn.ReLU()
        ]

        self._init(self.layers_features[0], np.sqrt(2))
        self._init(self.layers_features[2], np.sqrt(2))
        self._init(self.layers_features[4], np.sqrt(2))
        self._init(self.layers_features[7], np.sqrt(2))
        self._init(self.layers_features[9], np.sqrt(2))

        self.layers_value = [
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.feature_dim, 1)
        ]

        self._init(self.layers_value[0], 0.1)
        self._init(self.layers_value[2], 0.01)

        self.layers_policy = [
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU(),
        ]

        self._init(self.layers_policy[0], 0.01)

        if head == TYPE.discrete:
            self.layers_policy.append(DiscreteHead(self.feature_dim, action_dim))
        if head == TYPE.continuous:
            self.layers_policy.append(ContinuousHead(self.feature_dim, action_dim))
        if head == TYPE.multibinary:
            pass

        self.features = nn.Sequential(*self.layers_features)
        self.critic = nn.Sequential(*self.layers_value)
        self.actor = nn.Sequential(*self.layers_policy)

    def forward(self, state):
        features = self.features(state)
        value = self.critic(features)
        action, probs = self.actor(features)

        return value, action, probs

    def _init(self, layer, gain):
        nn.init.orthogonal_(layer.weight, gain)
        layer.bias.data.zero_()
