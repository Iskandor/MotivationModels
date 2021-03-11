import torch
from torch import nn
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
        input_height = self.input_shape[1]
        input_width = self.input_shape[2]
        self.feature_dim = 128 * (input_width // 16) * (input_height // 16)

        self.layers_features = [
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            nn.Flatten()
        ]

        self.layers_value = [
            torch.nn.Linear(self.feature_dim, config.critic_h1),
            torch.nn.ReLU(),
            torch.nn.Linear(config.critic_h1, config.critic_h2),
            torch.nn.ReLU(),
            torch.nn.Linear(config.critic_h2, 1)
        ]

        self.layers_policy = [
            torch.nn.Linear(self.feature_dim, config.actor_h1),
            torch.nn.ReLU(),
            torch.nn.Linear(config.actor_h1, config.actor_h2),
            torch.nn.ReLU(),
        ]

        if head == TYPE.discrete:
            self.layers_policy.append(DiscreteHead(config.actor_h2, action_dim))
        if head == TYPE.continuous:
            self.layers_policy.append(ContinuousHead(config.actor_h2, action_dim))
        if head == TYPE.multibinary:
            pass

        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_features[i].weight)

        for i in range(len(self.layers_value)):
            if hasattr(self.layers_value[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_value[i].weight)

        for i in range(len(self.layers_policy)):
            if hasattr(self.layers_policy[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_policy[i].weight)

        self.features = nn.Sequential(*self.layers_features)
        self.critic = nn.Sequential(*self.layers_value)
        self.actor = nn.Sequential(*self.layers_policy)

    def forward(self, state):
        features = self.features(state)
        value = self.critic(features)
        action, probs = self.actor(features)

        return value, action, probs


class SegaPPONetwork(torch.nn.Module):
    def __init__(self, input_shape, action_dim, action_space, config):
        super(SegaPPONetwork, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim
        input_channels = self.input_shape[2]
        input_height = self.input_shape[0]
        input_width = self.input_shape[1]
        self.feature_dim = 128 * (input_width // 16) * (input_height // 16)

        self.layers_features = [
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            nn.Flatten()
        ]

        self.layers_value = [
            torch.nn.Linear(self.feature_dim, config.critic_h1),
            torch.nn.ReLU(),
            torch.nn.Linear(config.critic_h1, config.critic_h2),
            torch.nn.ReLU(),
            torch.nn.Linear(config.critic_h2, 1)
        ]

        self.layers_policy = [
            torch.nn.Linear(self.feature_dim, config.actor_h1),
            torch.nn.ReLU(),
            torch.nn.Linear(config.actor_h1, config.actor_h2),
            torch.nn.ReLU(),
            torch.nn.Linear(config.actor_h2, action_dim),
        ]

        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_features[i].weight)

        for i in range(len(self.layers_value)):
            if hasattr(self.layers_value[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_value[i].weight)

        for i in range(len(self.layers_policy)):
            if hasattr(self.layers_policy[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_policy[i].weight)

        self.features = nn.Sequential(*self.layers_features)
        self.critic = nn.Sequential(*self.layers_value)
        self.actor = nn.Sequential(*self.layers_policy)

    def action(self, state):
        if state.ndim == 3:
            state = state.unsqueeze(0)
        features = self.features(state)
        policy = self.actor(features)
        return policy

    def value(self, state):
        if state.ndim == 3:
            state = state.unsqueeze(0)
        features = self.features(state)
        value = self.critic(features)
        return value
