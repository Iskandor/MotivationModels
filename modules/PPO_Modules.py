import torch
from torch import nn


class ContinuousPPONetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(ContinuousPPONetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.layers_value = [
            torch.nn.Linear(state_dim, config.critic_h1),
            torch.nn.ReLU(),
            torch.nn.Linear(config.critic_h1, config.critic_h2),
            torch.nn.ReLU(),
            torch.nn.Linear(config.critic_h2, 1)
        ]

        self.layers_policy = [
            torch.nn.Linear(state_dim, config.actor_h1),
            torch.nn.ReLU(),
            torch.nn.Linear(config.actor_h1, config.actor_h2),
            torch.nn.ReLU(),
        ]

        self.layers_mean = nn.Linear(config.actor_h2, action_dim)
        nn.init.uniform_(self.layers_mean.weight, -0.01, 0.01)
        self.layers_var = nn.Linear(config.actor_h2, action_dim)
        nn.init.uniform_(self.layers_var.weight, -1.0, 1.0)

        for i in range(len(self.layers_value)):
            if hasattr(self.layers_value[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_value[i].weight)

        for i in range(len(self.layers_policy)):
            if hasattr(self.layers_policy[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_policy[i].weight)

        self.critic = nn.Sequential(*self.layers_value)
        self.actor = nn.Sequential(*self.layers_policy)

    def dist(self, state):
        x = self.actor(state)
        mu = self.layers_mean(x)
        var = self.layers_var(x)
        return torch.distributions.Normal(mu, torch.exp(0.5 * var))

    def value(self, state):
        value = self.critic(state)
        return value


class AtariPPONetwork(torch.nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(AtariPPONetwork, self).__init__()

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
        self.features.to(config.device)

        self.critic = nn.Sequential(*self.layers_value)
        self.critic.to(config.device)

        self.actor = nn.Sequential(*self.layers_policy)
        self.actor.to(config.device)

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
        self.features.to(config.device)

        self.critic = nn.Sequential(*self.layers_value)
        self.critic.to(config.device)

        self.actor = nn.Sequential(*self.layers_policy)
        self.actor.to(config.device)

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
