import torch
import torch.nn as nn
import numpy as np

from agents import TYPE
from modules import init_orthogonal
from modules.PPO_Modules import Actor, DiscreteHead, Critic2Heads


class Aggregator(nn.Module):
    def __init__(self, n_env, state_dim, frequency):
        super(Aggregator, self).__init__()

        self.buffer = torch.zeros(frequency, n_env, state_dim)
        self.index = frequency - 1
        self.frequency = frequency

    def forward(self, state):
        aggregated_value = None

        self.buffer[self.index] = state
        self.index += 1

        if self.index == self.frequency:
            aggregated_value = self.buffer.mean(dim=0)
            self.buffer.zero_()

        return aggregated_value


class DOPControllerAtari(nn.Module):
    def __init__(self, state_dim, action_dim, config, features):
        super(DOPControllerAtari, self).__init__()

        self.features = features

        self.critic = nn.Sequential(
            torch.nn.Linear(state_dim, state_dim),
            torch.nn.ReLU(),
            Critic2Heads(state_dim)
        )

        init_orthogonal(self.critic[0], 0.1)
        init_orthogonal(self.critic[2], 0.01)

        self.actor = nn.Sequential(
            torch.nn.Linear(state_dim, state_dim),
            torch.nn.ReLU(),
            DiscreteHead(state_dim, action_dim)
        )

        init_orthogonal(self.actor[0], 0.01)
        init_orthogonal(self.actor[2], 0.01)

        self.actor = Actor(self.actor, TYPE.discrete, action_dim)

    def forward(self, state):
        features = self.features(state)
        value = self.critic(features)
        action, probs = self.actor(features)
        action = self.actor.encode_action(action)

        return value, action, probs


class DOPActorAtari2(nn.Module):
    def __init__(self, head_count, state_dim, action_dim, features, actor, critic):
        super(DOPActorAtari2, self).__init__()

        self.head_count = head_count
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.features = features
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        features = self.features(state)
        value = self.critic(features).view(-1, self.head_count, 2)
        action, probs = self.actor(features)
        action = self.actor.encode_action(action.view(-1, 1)).view(-1, self.head_count, self.action_dim)

        return value, action, probs

    def select_action(self, index, all_action, all_probs):
        index = index.argmax(dim=1, keepdim=True)
        all_action = all_action.view(-1, self.head_count, self.action_dim)
        all_probs = all_probs.view(-1, self.head_count, self.action_dim)
        action = all_action.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, self.action_dim)).squeeze(1)
        probs = all_probs.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, self.action_dim)).squeeze(1)
        return action, probs


class DOPActorAtari(nn.Module):
    def __init__(self, head_count, state_dim, action_dim, features, actor, critic):
        super(DOPActorAtari, self).__init__()

        self.head_count = head_count
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.features = features
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        index = state[:, 0:1].type(torch.int64)
        features = state[:, 1:]

        value = self.critic(features)
        all_action, all_probs = self.actor(features)

        all_action = self.actor.encode_action(all_action.view(-1, 1))
        action, probs = self.select_action(index, all_action, all_probs)

        return value, action, probs

    def select_action(self, index, all_action, all_probs):
        all_action = all_action.view(-1, self.head_count, self.action_dim)
        all_probs = all_probs.view(-1, self.head_count, self.action_dim)
        action = all_action.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, self.action_dim)).squeeze(1)
        probs = all_probs.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, self.action_dim)).squeeze(1)
        return action, probs


class DOPGeneratorAtari(nn.Module):
    def __init__(self, head_count, state_dim, action_dim, features, actor, critic):
        super(DOPGeneratorAtari, self).__init__()

        self.head_count = head_count
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.features = features
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        features = self.features(state)
        values = self.critic(features)
        action, probs = self.actor(features)
        action = self.actor.encode_action(action.view(-1, 1)).view(-1, self.head_count, self.action_dim)

        return values.unsqueeze(-1), action, probs


class DOPModelAtari(nn.Module):
    def __init__(self, head_count, state_dim, action_dim, config, features, actor, motivator):
        super(DOPModelAtari, self).__init__()

        self.head_count = head_count
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.motivator = motivator
        self.features = features
        self.actor = actor
        self.eta = config.motivation_eta

    def forward(self, state, action):
        predicted_code = self.motivator(state, action)
        return predicted_code

    def error(self, state, action):
        return self.motivator.error(state, action)

    def motivator_loss_function(self, state, action):
        return self.motivator.loss_function(state, action)

    def generator_loss_function(self, state):
        channels = state.shape[1]
        h = state.shape[2]
        w = state.shape[3]

        features = self.features(state)
        action, prob = self.actor(features)

        prob = prob.view(-1, self.action_dim)
        action = action.view(-1, 1)
        state = state.unsqueeze(1).repeat(1, self.head_count, 1, 1, 1).view(-1, channels, h, w)

        logprob = self.actor.log_prob(prob, action)
        action = self.actor.encode_action(action)
        error = self.error(state, action)

        loss = logprob * error * self.eta

        return -loss.mean()
