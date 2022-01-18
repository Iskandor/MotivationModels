import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from agents import TYPE
from modules import init_orthogonal, init_uniform, init_xavier_uniform, init_custom
from utils import one_hot_code


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
    def __init__(self, model, head, action_dim):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.head_type = head
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

    def encode_action(self, action):
        if self.head_type == TYPE.discrete:
            return one_hot_code(action, self.action_dim)
        if self.head_type == TYPE.continuous:
            return action
        if self.head_type == TYPE.multibinary:
            return None  # not implemented


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


class CriticHead(nn.Module):
    def __init__(self, input_dim, base, n_heads=1):
        super(CriticHead, self).__init__()
        self.base = base
        self.value = nn.Linear(input_dim, n_heads)

    def forward(self, x):
        x = self.base(x)
        value = self.value(x)
        return value

    @property
    def weight(self):
        return self.value.weight

    @property
    def bias(self):
        return self.value.bias


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

    @property
    def weight(self):
        return self.ext.weight, self.int.weight

    @property
    def bias(self):
        return self.ext.bias, self.int.bias


class Critic2NHeads(nn.Module):
    def __init__(self, input_dim, n_heads):
        super(Critic2NHeads, self).__init__()
        self.ext = nn.Linear(input_dim, n_heads)
        self.int = nn.Linear(input_dim, n_heads)

        init_orthogonal(self.ext, 0.01)
        init_orthogonal(self.int, 0.01)

    def forward(self, x):
        ext_value = self.ext(x)
        int_value = self.int(x)
        return torch.cat([ext_value, int_value], dim=1).squeeze(-1)

    @property
    def weight(self):
        return self.ext.weight, self.int.weight

    @property
    def bias(self):
        return self.ext.bias, self.int.bias


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

        self.layers_actor = nn.Sequential(
            torch.nn.Linear(state_dim, config.actor_h1),
            nn.ReLU(),
            torch.nn.Linear(config.actor_h1, config.actor_h2),
            nn.ReLU(),
            DiscreteHead(config.actor_h2, action_dim)
        )
        nn.init.xavier_uniform_(self.layers_actor[0].weight)
        nn.init.xavier_uniform_(self.layers_actor[2].weight)

        self.actor = Actor(self.layers_actor, head, action_dim)

    def forward(self, state):
        value = self.critic(state)
        action, probs = self.actor(state)

        return value, action, probs
