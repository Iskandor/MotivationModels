import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical, MultivariateNormal, Normal

from agents import TYPE
from modules import init
from modules.forward_models.ForwardModelAtari import ForwardModelAtari
from modules.rnd_models.RNDModelAeris import RNDModelAeris, QRNDModelAeris, DOPModelAeris
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

        nn.init.orthogonal_(self.mu[0].weight, 0.01)
        nn.init.zeros_(self.mu[0].bias)
        nn.init.orthogonal_(self.var[0].weight, 1)
        nn.init.zeros_(self.var[0].bias)

        self.action_dim = action_dim

    def forward(self, x):
        mu = self.mu(x)
        var = self.var(x)

        dist = Normal(mu, var.sqrt())
        action = dist.sample()

        return action, torch.cat([mu, var], dim=1)

    @staticmethod
    def log_prob(probs, actions):
        dim = probs.shape[1]
        mu, var = probs[:, :dim // 2], probs[:, dim // 2:]

        p1 = -((actions - mu) ** 2) / (2.0 * var + 0.1)
        p2 = -torch.log(torch.sqrt(2.0 * np.pi * var))

        log_prob = p1 + p2

        return log_prob

    @staticmethod
    def entropy(probs):
        dim = probs.shape[1]
        var = probs[:, dim // 2:]
        entropy = -(torch.log(2.0 * np.pi * var) + 1.0) / 2.0

        return entropy.mean()


class Actor(nn.Module):
    def __init__(self, action_dim, layers, head):
        super(Actor, self).__init__()
        self.head = None
        if head == TYPE.discrete:
            self.head = DiscreteHead
        if head == TYPE.continuous:
            self.head = ContinuousHead
        if head == TYPE.multibinary:
            pass

        layers.append(self.head(layers[-2].out_features, action_dim))
        self.actor = nn.Sequential(*layers)

    def forward(self, x):
        return self.actor(x)

    def log_prob(self, probs, actions):
        return self.head.log_prob(probs, actions)

    def entropy(self, probs):
        return self.head.entropy(probs)


class ActorNHeads(nn.Module):
    def __init__(self, head_count, action_dim, layers, head, config):
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
        self.heads = [nn.Sequential(
            nn.Linear(self.actor[-2].out_features, config.actor_h1),
            nn.ReLU(),
            self.head(config.actor_h1, action_dim))
            for _ in range(head_count)]

        for h in self.heads:
            nn.init.xavier_uniform_(h[0].weight)
            nn.init.zeros_(h[0].bias)

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
        return torch.cat([ext_value, int_value], dim=1).squeeze(-1)


class Residual(torch.nn.Module):
    def __init__(self, features):
        super(Residual, self).__init__()

        self.layer = nn.Linear(features, features)
        init(self.layer, 0.01)

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
            nn.ReLU()
        ]
        nn.init.xavier_uniform_(self.layers_actor[0].weight)
        nn.init.xavier_uniform_(self.layers_actor[2].weight)

        self.actor = Actor(action_dim, self.layers_actor, head)

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

        # self.features = nn.Sequential(
        #     nn.Conv1d(self.channels, config.critic_kernels_count, kernel_size=8, stride=4, padding=2),
        #     nn.ReLU(),
        #     nn.Conv1d(config.critic_kernels_count, config.critic_kernels_count * 2, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(config.critic_kernels_count * 2, config.critic_kernels_count * 2, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(fc_count, fc_count),
        #     nn.ReLU()
        # )

        # init(self.features[0], np.sqrt(2))
        # init(self.features[2], np.sqrt(2))
        # init(self.features[4], np.sqrt(2))
        # init(self.features[7], np.sqrt(2))

        self.critic = nn.Sequential(
            nn.Conv1d(self.channels, config.critic_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(config.critic_kernels_count, config.critic_kernels_count * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(config.critic_kernels_count * 2, config.critic_kernels_count * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_count, fc_count),
            nn.ReLU(),
            nn.Linear(fc_count, config.critic_h1),
            nn.ReLU(),
            nn.Linear(config.critic_h1, 1))

        init(self.critic[0], 1)
        init(self.critic[2], 1)
        init(self.critic[4], 1)
        init(self.critic[7], 1)
        init(self.critic[9], 0.01)
        init(self.critic[11], 0.01)

        self.channels = input_shape[0]
        self.width = input_shape[1]

        self.layers_actor = [
            nn.Conv1d(self.channels, config.critic_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(config.critic_kernels_count, config.critic_kernels_count * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(config.critic_kernels_count * 2, config.critic_kernels_count * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_count, fc_count),
            nn.ReLU(),
            nn.Linear(fc_count, config.actor_h1),
            nn.ReLU()]

        init(self.layers_actor[0], 1)
        init(self.layers_actor[2], 1)
        init(self.layers_actor[4], 1)
        init(self.layers_actor[7], 1)
        init(self.layers_actor[9], 0.01)

        self.actor = Actor(action_dim, self.layers_actor, head)

    def forward(self, state):
        # x = self.features(state)
        x = state
        value = self.critic(x)
        action, probs = self.actor(x)

        return value, action, probs


class PPOAerisMotivationNetwork(PPOAerisNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAerisMotivationNetwork, self).__init__(input_shape, action_dim, config, head)

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
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAerisNetworkRND, self).__init__(input_shape, action_dim, config, head)
        self.rnd_model = RNDModelAeris(input_shape, action_dim, config)


class PPOAerisNetworkDOP(PPOAerisNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAerisNetworkDOP, self).__init__(input_shape, action_dim, config, head)

        self.action_dim = action_dim
        self.channels = input_shape[0]
        self.width = input_shape[1]
        self.head_count = 4

        fc_count = config.critic_kernels_count * self.width // 4

        self.layers_actor = [
            nn.Linear(fc_count, fc_count),
            nn.ReLU()]

        nn.init.xavier_uniform_(self.layers_actor[0].weight)
        nn.init.zeros_(self.layers_actor[0].bias)

        self.actor = ActorNHeads(self.head_count, action_dim, self.layers_actor, head, config)
        self.motivator = QRNDModelAeris(input_shape, action_dim * 2, config)
        self.dop_model = DOPModelAeris(input_shape, action_dim * 2, config, self.features, self.actor, self.motivator)
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
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAerisNetworkDOPRef, self).__init__(input_shape, action_dim, config, head)

        self.action_dim = action_dim
        self.channels = input_shape[0]
        self.width = input_shape[1]
        self.head_count = 4

        fc_count = config.critic_kernels_count * self.width // 4

        self.layers_actor = [
            nn.Linear(fc_count, fc_count),
            nn.ReLU()]

        nn.init.xavier_uniform_(self.layers_actor[0].weight)
        nn.init.zeros_(self.layers_actor[0].bias)

        self.actor = ActorNHeads(self.head_count, action_dim, self.layers_actor, head, config)
        self.indices = []

    def forward(self, state):
        x = self.features(state)
        value = self.critic(x)
        action, probs = self.actor(x)

        argmax = torch.randint(0, self.head_count, (state.shape[0],))
        action = action.gather(dim=1, index=argmax.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.action_dim))
        probs = probs.gather(dim=1, index=argmax.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.action_dim * 2))

        return value, action.squeeze(1), probs.squeeze(1)


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

        init(self.layers_features[0], np.sqrt(2))
        init(self.layers_features[2], np.sqrt(2))
        init(self.layers_features[4], np.sqrt(2))
        init(self.layers_features[6], np.sqrt(2))
        init(self.layers_features[9], 0.01)
        init(self.layers_features[11], 0.01)

        self.layers_value = [
            Residual(self.feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.feature_dim, 1)
        ]

        init(self.layers_value[2], 0.01)

        self.layers_policy = [
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU(),
        ]

        init(self.layers_policy[0], 0.01)

        self.features = nn.Sequential(*self.layers_features)
        self.critic = nn.Sequential(*self.layers_value)
        self.actor = Actor(action_dim, self.layers_policy, head)

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
