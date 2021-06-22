import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical, MultivariateNormal, Normal

from agents import TYPE
from modules import init
from modules.forward_models.ForwardModelAtari import ForwardModelAtari
from modules.rnd_models.RNDModelAeris import RNDModelAeris, DOPSimpleModelAeris, QRNDModelAeris, DOPModelAeris


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

        return action, torch.stack([mu, var], dim=1)

    @staticmethod
    def log_prob(probs, actions):
        mu, var = probs[:, 0], probs[:, 1]
        dist = Normal(mu, var.sqrt())
        log_prob = dist.log_prob(actions)

        return log_prob

    @staticmethod
    def entropy(probs):
        mu, var = probs[:, 0], probs[:, 1]
        dist = Normal(mu, var.sqrt())
        entropy = -dist.entropy()

        return entropy.mean()


class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, layers, head):
        super(Actor, self).__init__()
        self.head = None
        if head == TYPE.discrete:
            self.head = DiscreteHead
        if head == TYPE.continuous:
            self.head = ContinuousHead
        if head == TYPE.multibinary:
            pass

        layers.append(self.head(input_dim, action_dim))
        self.actor = nn.Sequential(*layers)

    def forward(self, x):
        return self.actor(x)

    def log_prob(self, probs, actions):
        return self.head.log_prob(probs, actions)

    def entropy(self, probs):
        return self.head.entropy(probs)


class ActorNHeads(nn.Module):
    def __init__(self, head_count, input_dim, action_dim, layers, head):
        super(ActorNHeads, self).__init__()
        self.head = None
        if head == TYPE.discrete:
            self.head = DiscreteHead
        if head == TYPE.continuous:
            self.head = ContinuousHead
        if head == TYPE.multibinary:
            pass

        self.actor = nn.Sequential(*layers)
        self.head_count = head_count
        self.heads = [self.head(input_dim, action_dim) for _ in range(head_count)]

    def forward(self, x):
        x = self.actor(x)
        probs = []
        actions = []

        for h in self.heads:
            a, p = h(x)
            actions.append(a)
            probs.append(p)

        return torch.stack(actions, dim=1), torch.stack(probs, dim=1)

    def log_prob(self, probs, actions):
        return self.head.log_prob(probs, actions)

    def entropy(self, probs):
        return self.head.entropy(probs)


class Critic2Heads(nn.Module):
    def __init__(self, input_dim):
        super(Critic2Heads, self).__init__()
        self.ext = nn.Linear(input_dim, 1)
        self.int = nn.Linear(input_dim, 1)

        init(self.ext, 0.01)
        init(self.int, 0.01)

    def forward(self, x):
        ext_value = self.ext(x)
        int_value = self.int(x)
        return torch.stack([ext_value, int_value], dim=1).squeeze(-1)


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

        self.actor = Actor(config.actor_h2, action_dim, self.layers_actor, head)

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

        self.features = nn.Sequential(
            nn.Conv1d(self.channels, config.critic_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ELU(),
            nn.Conv1d(config.critic_kernels_count, config.critic_kernels_count * 2, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv1d(config.critic_kernels_count * 2, config.critic_kernels_count * 2, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Flatten()
        )

        nn.init.orthogonal_(self.features[0].weight)
        nn.init.orthogonal_(self.features[2].weight)
        nn.init.orthogonal_(self.features[4].weight)

        self.critic = nn.Sequential(
            nn.Linear(fc_count, fc_count),
            nn.ReLU(),
            nn.Linear(fc_count, config.critic_h1),
            nn.ReLU(),
            nn.Linear(config.critic_h1, 1))

        nn.init.xavier_uniform_(self.critic[0].weight)
        nn.init.xavier_uniform_(self.critic[2].weight)
        nn.init.uniform_(self.critic[4].weight, -0.003, 0.003)

        self.channels = input_shape[0]
        self.width = input_shape[1]

        self.layers_actor = [
            nn.Linear(fc_count, fc_count),
            nn.ReLU(),
            nn.Linear(fc_count, config.actor_h1),
            nn.ReLU()]

        nn.init.xavier_uniform_(self.layers_actor[0].weight)
        nn.init.xavier_uniform_(self.layers_actor[2].weight)

        self.actor = Actor(config.actor_h1, action_dim, self.layers_actor, head)

    def forward(self, state):
        x = self.features(state)
        value = self.critic(x)
        action, probs = self.actor(x)

        return value, action, probs


class PPOAerisMotivationNetwork(torch.nn.Module):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAerisMotivationNetwork, self).__init__()

        self.channels = input_shape[0]
        self.width = input_shape[1]

        fc_count = config.critic_kernels_count * self.width // 4

        self.features = nn.Sequential(
            nn.Conv1d(self.channels, config.critic_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ELU(),
            nn.Conv1d(config.critic_kernels_count, config.critic_kernels_count * 2, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv1d(config.critic_kernels_count * 2, config.critic_kernels_count * 2, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Flatten()
        )

        nn.init.orthogonal_(self.features[0].weight)
        nn.init.orthogonal_(self.features[2].weight)
        nn.init.orthogonal_(self.features[4].weight)

        self.critic = nn.Sequential(
            nn.Linear(fc_count, config.critic_h1),
            nn.ReLU(),
            Critic2Heads(config.critic_h1))

        nn.init.xavier_uniform_(self.critic[0].weight)

        self.channels = input_shape[0]
        self.width = input_shape[1]

        self.layers_actor = [
            nn.Linear(fc_count, config.actor_h1),
            nn.ReLU()]

        nn.init.xavier_uniform_(self.layers_actor[0].weight)

        self.actor = Actor(config.actor_h1, action_dim, self.layers_actor, head)

    def forward(self, state):
        x = self.features(state)
        value = self.critic(x)
        action, probs = self.actor(x)

        return value, action, probs


class PPOAerisNetworkRND(PPOAerisMotivationNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAerisNetworkRND, self).__init__(input_shape, action_dim, config, head)
        self.rnd_model = RNDModelAeris(input_shape, action_dim, config)


class PPOAerisNetworkDOPSimple(PPOAerisNetwork):
    def __init__(self, state_dim, action_dim, config, head):
        super(PPOAerisNetworkDOPSimple, self).__init__(state_dim, action_dim, config, head)
        self.dop_model = DOPSimpleModelAeris(state_dim, action_dim, config, self.features, self.actor)


class PPOAerisNetworkDOP(PPOAerisNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAerisNetworkDOP, self).__init__(input_shape, action_dim, config, head)

        self.action_dim = action_dim
        self.channels = input_shape[0]
        self.width = input_shape[1]
        self.head_count = 16

        fc_count = config.critic_kernels_count * self.width // 4

        self.layers_actor = [
            nn.Linear(fc_count, fc_count),
            nn.ReLU(),
            nn.Linear(fc_count, config.actor_h1),
            nn.ReLU()]

        nn.init.xavier_uniform_(self.layers_actor[0].weight)
        nn.init.xavier_uniform_(self.layers_actor[2].weight)

        self.actor = ActorNHeads(self.head_count, config.actor_h1, action_dim, self.layers_actor, head)
        self.motivator = QRNDModelAeris(input_shape, action_dim, config)
        self.dop_model = DOPModelAeris(input_shape, action_dim, config, self.features, self.actor, self.motivator)

    def forward(self, state):
        x = self.features(state)
        value = self.critic(x)
        action, probs = self.actor(x)

        state = state.unsqueeze(1).repeat(1, self.head_count, 1, 1).view(-1, self.channels, self.width)
        action = action.view(-1, self.action_dim)

        error = self.motivator.error(state, action).view(-1, self.head_count)
        argmax = error.argmax(dim=1)
        action = action.view(-1, self.head_count, self.action_dim)
        action = action.gather(dim=1, index=argmax.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.action_dim))
        probs = probs.gather(dim=1, index=argmax.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.action_dim, 2))

        return value, action.squeeze(1), probs.squeeze(1)


class PPOAtariNetwork(torch.nn.Module):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAtariNetwork, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim
        input_channels = self.input_shape[0]
        self.feature_dim = 512

        self.layers_features = [
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(12 * 12 * 64, self.feature_dim),
            nn.ReLU()
        ]

        init(self.layers_features[0], np.sqrt(2))
        init(self.layers_features[2], np.sqrt(2))
        init(self.layers_features[4], np.sqrt(2))
        init(self.layers_features[7], np.sqrt(2))

        self.layers_value = [
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.feature_dim, 1)
        ]

        init(self.layers_value[0], 0.1)
        init(self.layers_value[2], 0.01)

        self.layers_policy = [
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU(),
        ]

        init(self.layers_policy[0], 0.01)

        self.features = nn.Sequential(*self.layers_features)
        self.critic = nn.Sequential(*self.layers_value)
        self.actor = Actor(self.feature_dim, action_dim, self.layers_actor, head)

    def forward(self, state):
        features = self.features(state)
        value = self.critic(features)
        action, probs = self.actor(features)

        return value, action, probs


class PPOAtariMotivationNetwork(torch.nn.Module):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAtariMotivationNetwork, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim
        input_channels = self.input_shape[0]
        self.feature_dim = 512

        self.layers_features = [
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(12 * 12 * 64, self.feature_dim),
            nn.ReLU()
        ]

        init(self.layers_features[0], np.sqrt(2))
        init(self.layers_features[2], np.sqrt(2))
        init(self.layers_features[4], np.sqrt(2))
        init(self.layers_features[7], np.sqrt(2))

        self.layers_value = [
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU(),
            Critic2Heads(self.feature_dim)
        ]

        init(self.layers_value[0], 0.1)

        self.layers_policy = [
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU(),
        ]

        init(self.layers_policy[0], 0.01)

        self.features = nn.Sequential(*self.layers_features)
        self.critic = nn.Sequential(*self.layers_value)
        self.actor = Actor(self.feature_dim, action_dim, self.layers_actor, head)

    def forward(self, state):
        features = self.features(state)
        value = self.critic(features)
        action, probs = self.actor(features)

        return value, action, probs


class PPOAtariNetworkFM(PPOAtariMotivationNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAtariNetworkFM, self).__init__(input_shape, action_dim, config, head)
        self.forward_model = ForwardModelAtari(self.features, self.feature_dim, self.action_dim)
