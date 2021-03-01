import copy

import torch
from torch import nn
from torch.nn import *


class EncoderRobotic(nn.Module):
    def __init__(self, input_shape, feature_dim):
        super(EncoderRobotic, self).__init__()
        self.layers_encoder = [
            Linear(in_features=input_shape, out_features=input_shape * 10, bias=True),
            LeakyReLU(),
            Linear(in_features=input_shape * 10, out_features=input_shape * 10, bias=True),
            LeakyReLU(),
            Linear(in_features=input_shape * 10, out_features=input_shape * 5, bias=True),
            LeakyReLU(),
            Linear(in_features=input_shape * 5, out_features=input_shape * 5, bias=True),
            LeakyReLU(),
            Linear(in_features=input_shape * 5, out_features=feature_dim, bias=True)
        ]

        nn.init.xavier_uniform_(self.layers_encoder[0].weight)
        nn.init.xavier_uniform_(self.layers_encoder[2].weight)
        nn.init.xavier_uniform_(self.layers_encoder[4].weight)
        nn.init.xavier_uniform_(self.layers_encoder[6].weight)
        nn.init.xavier_uniform_(self.layers_encoder[8].weight)

        self.encoder = Sequential(*self.layers_encoder)

    def forward(self, state):
        x = self.encoder(state)
        return x

    def loss_function(self, state, next_state):
        loss = self.variation_prior(state) + self.stability_prior(state, next_state)
        return loss

    def variation_prior(self, state):
        sa = state[torch.randperm(state.shape[0])]
        sb = state[torch.randperm(state.shape[0])]
        variation_loss = torch.exp((self.encoder(sa) - self.encoder(sb)).abs() * -1.0).mean()
        return variation_loss

    def stability_prior(self, state, next_state):
        stability_loss = (self.encoder(next_state) - self.encoder(state)).abs().pow(2).mean()
        return stability_loss

    def distance_conservation(self, state):
        sa = state[torch.randperm(state.shape[0])]
        sb = state[torch.randperm(state.shape[0])]
        esa = self.encoder(sa)
        esb = self.encoder(sb)
        d1 = (sa - sb).pow(2).sum().sqrt()
        d2 = (esa - esb).pow(2).sum().sqrt()

        distance_conservation_loss = nn.functional.mse_loss(d2, d1)
        return distance_conservation_loss


class CriticRobotic(nn.Module):
    def __init__(self, feature_dim, action_dim, config):
        super(CriticRobotic, self).__init__()

        self._hidden0 = nn.Linear(feature_dim, config.critic_h1)
        self._hidden1 = nn.Linear(config.critic_h1 + action_dim, config.critic_h2)
        self._output = nn.Linear(config.critic_h2, 1)

        self.init()
        self.heads = 1

    def forward(self, features, action):
        x = torch.relu(self._hidden0(features))
        x = torch.cat([x, action], 1)
        x = torch.relu(self._hidden1(x))
        value = self._output(x)
        return value

    def init(self):
        nn.init.xavier_uniform_(self._hidden0.weight)
        nn.init.xavier_uniform_(self._hidden1.weight)
        nn.init.uniform_(self._output.weight, -3e-3, 3e-3)


class ActorRobotic(nn.Module):
    def __init__(self, feature_dim, action_dim, config):
        super(ActorRobotic, self).__init__()

        self._hidden0 = nn.Linear(feature_dim, config.actor_h1)
        self._hidden1 = nn.Linear(config.actor_h1, config.actor_h2)
        self._output = nn.Linear(config.actor_h2, action_dim)

        self.init()

    def forward(self, features):
        x = torch.relu(self._hidden0(features))
        x = torch.relu(self._hidden1(x))
        policy = torch.tanh(self._output(x))
        return policy

    def init(self):
        nn.init.xavier_uniform_(self._hidden0.weight)
        nn.init.xavier_uniform_(self._hidden1.weight)
        nn.init.uniform_(self._output.weight, -3e-1, 3e-1)


class BaselineRobotic(nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(BaselineRobotic, self).__init__()
        feature_dim = action_dim
        self.encoder = EncoderRobotic(input_shape, feature_dim)
        self.critic = CriticRobotic(feature_dim, action_dim, config)
        self.actor = ActorRobotic(feature_dim, action_dim, config)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
        self.hard_update()

    def value(self, state, action):
        features = self.encoder(state)
        value = self.critic(features, action)

        return value

    def action(self, state):
        features = self.encoder(state)
        policy = self.actor(features)
        return policy

    def value_target(self, state, action):
        features = self.encoder(state)
        value = self.critic_target(features, action)

        return value

    def action_target(self, state):
        features = self.encoder(state)
        policy = self.actor_target(features)
        return policy

    def loss_function(self, state, action, next_state, expected_values):
        features = self.encoder(state)
        loss = torch.nn.functional.mse_loss(self.critic(features, action), expected_values) * 2 - self.critic(features, self.actor(features)).mean() + self.encoder.loss_function(state, next_state)
        return loss

    def soft_update(self, tau):
        self._soft_update(self.critic_target, self.critic, tau)
        self._soft_update(self.actor_target, self.actor, tau)

    def hard_update(self):
        self._hard_update(self.critic_target, self.critic)
        self._hard_update(self.actor_target, self.actor)

    @staticmethod
    def _soft_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    @staticmethod
    def _hard_update(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)




