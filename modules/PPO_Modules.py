import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from agents import TYPE
from modules import init_orthogonal, init_uniform, init_xavier_uniform, init_custom
from modules.forward_models.ForwardModelAtari import ForwardModelAtari
from modules.rnd_models.RNDModelAtari import RNDModelAtari


class DiscreteHead(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DiscreteHead, self).__init__()
        self.logits = nn.Linear(input_dim, action_dim, bias=True)

        torch.nn.init.xavier_uniform_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    def forward(self, x):
        logits = self.logits(x)
        probs = torch.softmax(logits, dim=1)
        dist = Categorical(probs)

        action = dist.sample().unsqueeze(1)

        return action, probs

    @staticmethod
    def log_prob(probs, actions):
        actions = torch.argmax(actions, dim=1)
        dist = Categorical(probs)
        log_prob = dist.log_prob(actions).unsqueeze(1)

        return log_prob

    @staticmethod
    def entropy(probs):
        dist = Categorical(probs)
        entropy = -dist.entropy()
        return entropy.mean()

    @property
    def weight(self):
        return self.logits.weight

    @property
    def bias(self):
        return self.logits.bias


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

        init_uniform(self.mu[0], 0.03)
        init_uniform(self.var[0], 0.03)

        self.action_dim = action_dim

    def forward(self, x):
        mu = self.mu(x)
        var = self.var(x)

        dist = Normal(mu, var.sqrt().clamp(min=1e-3))
        action = dist.sample()

        return action, torch.cat([mu, var], dim=1)

    @staticmethod
    def log_prob(probs, actions):
        dim = probs.shape[1]
        mu, var = probs[:, :dim // 2], probs[:, dim // 2:]

        p1 = - ((actions - mu) ** 2) / (2.0 * var.clamp(min=1e-3))
        p2 = - torch.log(torch.sqrt(2.0 * np.pi * var))

        log_prob = p1 + p2

        return log_prob

    @staticmethod
    def entropy(probs):
        dim = probs.shape[1]
        var = probs[:, dim // 2:]
        entropy = -(torch.log(2.0 * np.pi * var) + 1.0) / 2.0

        return entropy.mean()


class Actor(nn.Module):
    def __init__(self, model, head):
        super(Actor, self).__init__()
        self.head = None
        if head == TYPE.discrete:
            self.head = DiscreteHead
        if head == TYPE.continuous:
            self.head = ContinuousHead
        if head == TYPE.multibinary:
            pass

        self.model = model

    def forward(self, x):
        return self.model(x)

    def log_prob(self, probs, actions):
        return self.head.log_prob(probs, actions)

    def entropy(self, probs):
        return self.head.entropy(probs)


class ActorNHeads(nn.Module):
    def __init__(self, head, head_count, dims, init='orto'):
        super(ActorNHeads, self).__init__()

        self.head = None
        if head == TYPE.discrete:
            self.head = DiscreteHead
        if head == TYPE.continuous:
            self.head = ContinuousHead
        if head == TYPE.multibinary:
            pass

        self.heads = nn.ModuleList([nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            self.head(dims[1], dims[2]))
            for _ in range(head_count)])

        if init == 'xavier':
            self.xavier_init()
        if init == 'orto':
            self.orthogonal_init(head_count, dims)

    def xavier_init(self):
        for i, h in enumerate(self.heads):
            init_xavier_uniform(h[0])

    def orthogonal_init(self, head_count, dims):
        weight1 = torch.zeros(head_count * dims[1], dims[0])
        nn.init.orthogonal_(weight1, 1)
        weight1 = weight1.reshape(head_count, dims[1], dims[0])

        for i, h in enumerate(self.heads):
            init_custom(h[0], weight1[i])

    def forward(self, x):
        actions = []
        probs = []

        for h in self.heads:
            a, p = h(x)
            actions.append(a)
            probs.append(p)

        return torch.stack(actions, dim=1), torch.stack(probs, dim=1)


class Critic2Heads(nn.Module):
    def __init__(self, input_dim):
        super(Critic2Heads, self).__init__()
        self.ext = nn.Linear(input_dim, 1)
        self.int = nn.Linear(input_dim, 1)

        init_orthogonal(self.ext, 0.01)
        init_orthogonal(self.int, 0.01)

    def forward(self, x):
        ext_value = self.ext(x)
        int_value = self.int(x)
        return torch.cat([ext_value, int_value], dim=1).squeeze(-1)


class Residual(torch.nn.Module):
    def __init__(self, features):
        super(Residual, self).__init__()

        self.layer = nn.Linear(features, features)
        init_orthogonal(self.layer, 0.01)

    def forward(self, x):
        return x + self.layer(x)


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
            nn.ReLU(),
        ]
        nn.init.xavier_uniform_(self.layers_actor[0].weight)
        nn.init.xavier_uniform_(self.layers_actor[2].weight)

        self.actor = Actor(self.layers_actor, head)

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
        self.feature_dim = 448

        fc_inputs_count = 64 * (input_width // 8) * (input_height // 8)

        self.layers_features = [
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_inputs_count, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU()
        ]

        init_orthogonal(self.layers_features[0], np.sqrt(2))
        init_orthogonal(self.layers_features[2], np.sqrt(2))
        init_orthogonal(self.layers_features[4], np.sqrt(2))
        init_orthogonal(self.layers_features[6], np.sqrt(2))
        init_orthogonal(self.layers_features[9], 0.01)
        init_orthogonal(self.layers_features[11], 0.01)

        self.layers_value = [
            Residual(self.feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.feature_dim, 1)
        ]

        init_orthogonal(self.layers_value[2], 0.01)

        self.layers_policy = nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU(),
            DiscreteHead(self.feature_dim, action_dim)
        )

        init_orthogonal(self.layers_policy[0], 0.01)
        init_orthogonal(self.layers_policy[2], 0.01)

        self.features = nn.Sequential(*self.layers_features)
        self.critic = nn.Sequential(*self.layers_value)
        self.actor = Actor(self.layers_policy, head)

    def forward(self, state):
        features = self.features(state)
        value = self.critic(features)
        action, probs = self.actor(features)

        return value, action, probs


class PPOAtariMotivationNetwork(PPOAtariNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAtariMotivationNetwork, self).__init__(input_shape, action_dim, config, head)

        self.layers_value = [
            Residual(self.feature_dim),
            torch.nn.ReLU(),
            Critic2Heads(self.feature_dim)
        ]

        self.critic = nn.Sequential(*self.layers_value)


class PPOAtariNetworkFM(PPOAtariMotivationNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAtariNetworkFM, self).__init__(input_shape, action_dim, config, head)
        self.forward_model = ForwardModelAtari(self.features, self.feature_dim, self.action_dim)


class PPOAtariNetworkRND(PPOAtariMotivationNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAtariNetworkRND, self).__init__(input_shape, action_dim, config, head)
        self.rnd_model = RNDModelAtari(input_shape, self.action_dim, config)
