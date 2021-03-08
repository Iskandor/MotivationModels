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

        action = dist.sample().unsqueeze(0)

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


class PPONetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, config, head):
        super(PPONetwork, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(state_dim, config.critic_h1),
            nn.ReLU(),
            nn.Linear(config.critic_h1, config.critic_h2),
            nn.ReLU(),
            torch.nn.Linear(config.critic_h2, 1)
        )
        nn.init.xavier_uniform_(self.critic[0].weight)
        nn.init.xavier_uniform_(self.critic[2].weight)

        self.actor = nn.Sequential(
            torch.nn.Linear(state_dim, config.actor_h1),
            nn.ReLU(),
            torch.nn.Linear(config.actor_h1, config.actor_h2),
            nn.ReLU()
        )
        nn.init.xavier_uniform_(self.actor[0].weight)
        nn.init.xavier_uniform_(self.actor[2].weight)

        self.actor_head = None

        if head == TYPE.discrete:
            self.actor_head = DiscreteHead(config.actor_h2, action_dim)
        if head == TYPE.continuous:
            self.actor_head = ContinuousHead(config.actor_h2, action_dim)
        if head == TYPE.multibinary:
            pass

    def forward(self, state):
        value = self.critic(state)
        action, probs = self.actor_head(self.actor(state))

        return value, action, probs


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

        self.dist = Categorical

    def value(self, state):
        features = self.features(state)
        value = self.critic(features)
        return value

    def action(self, state, deterministic=False):
        features = self.features(state)
        logits = self.actor(features)
        probs = torch.softmax(logits, dim=1)
        dist = self.dist(probs)
        if deterministic:
            a = probs.argmax()
        else:
            a = dist.sample()

        l = dist.log_prob(a).detach()

        return a, l

    def evaluate(self, state, action):
        features = self.features(state)
        logits = self.actor(features)
        probs = torch.softmax(logits, dim=1)
        dist = self.dist(probs)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()

        return log_prob, entropy


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
