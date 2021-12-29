import torch
import torch.nn as nn
import numpy as np

from agents import TYPE
from modules import init_orthogonal
from modules.PPO_Modules import Actor, DiscreteHead, Critic2Heads


class DOPControllerAtari(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(DOPControllerAtari, self).__init__()

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
        value = self.critic(state)
        action, probs = self.actor(state)
        action = self.actor.encode_action(action)

        return value, action, probs


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
