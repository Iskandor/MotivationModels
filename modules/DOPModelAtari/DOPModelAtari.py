import torch
import torch.nn as nn
import numpy as np


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
