import torch
import torch.nn as nn
import numpy as np


class MetaCriticModelAeris(nn.Module):
    def __init__(self, forward_model, input_shape, action_dim):
        super(MetaCriticModelAeris, self).__init__()

        self.forward_model = forward_model
        self.input_shape = input_shape
        self.action_dim = action_dim

        self.layers = [
            nn.Flatten(),
            nn.LeakyReLU(),
            nn.Linear(input_shape[0] * input_shape[1], input_shape[0] * input_shape[1] // 2),
            nn.LeakyReLU(),
            nn.Linear(input_shape[0] * input_shape[1] // 2, 1)
        ]

        nn.init.xavier_uniform_(self.layers[2].weight)
        nn.init.uniform_(self.layers[4].weight, -0.003, 0.003)

        self.model = nn.Sequential(*self.layers)

    def forward(self, state, action):
        predicted_state = self.forward_model(state, action)
        error_estimate = self.model(predicted_state)

        return predicted_state, error_estimate

    def error(self, state, action, next_state):
        with torch.no_grad():
            predicted_state, error_estimate = self(state, action)
            error = self._error(predicted_state, next_state)

        return error, error_estimate

    def _error(self, predicted_state, next_state):
        error = torch.mean(torch.pow(predicted_state.view(predicted_state.shape[0], -1) - next_state.view(next_state.shape[0], -1), 2), dim=1).unsqueeze(1)
        return error

    def loss_function(self, state, action, next_state):
        predicted_state, error_estimate = self(state, action)
        error = self._error(predicted_state, next_state).detach()
        loss = nn.functional.mse_loss(predicted_state, next_state) + nn.functional.mse_loss(error_estimate, error)
        return loss


class MetaCriticRNDModelAeris(nn.Module):
    def __init__(self, rnd_model, input_shape, action_dim, config):
        super(MetaCriticRNDModelAeris, self).__init__()

        self.rnd_model = rnd_model
        self.input_shape = input_shape
        self.action_dim = action_dim

        self.layers = [
            nn.ReLU(),
            nn.Linear(config.metacritic_h1, config.metacritic_h1),
            nn.ReLU(),
            nn.Linear(config.metacritic_h1, config.metacritic_h1),
            nn.ReLU(),
            nn.Linear(config.metacritic_h1, 1)
        ]

        self._init(self.layers[1], np.sqrt(2))
        self._init(self.layers[3], np.sqrt(2))
        self._init(self.layers[5], np.sqrt(2))

        self.model = nn.Sequential(*self.layers)

    def forward(self, state):
        predicted_state = self.rnd_model(state)
        error_estimate = self.model(predicted_state)

        return predicted_state, error_estimate

    def error(self, state):
        with torch.no_grad():
            predicted_state, error_estimate = self(state)
            error = self._error(predicted_state, state)

        return error, error_estimate

    def _error(self, predicted_state, state):
        target = self.rnd_model.encode(state)
        error = torch.mean(torch.pow(predicted_state.view(predicted_state.shape[0], -1) - target.view(target.shape[0], -1), 2), dim=1).unsqueeze(1)
        return error

    def loss_function(self, state):
        predicted_state, error_estimate = self(state)
        error = self._error(predicted_state, state).detach()
        loss = self.rnd_model.loss_function(state, predicted_state) + nn.functional.mse_loss(error_estimate, error)
        return loss

    def _init(self, layer, gain):
        nn.init.orthogonal_(layer.weight, gain)
        layer.bias.data.zero_()
