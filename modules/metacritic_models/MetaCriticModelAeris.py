import torch
from torch import nn
from torch.nn import *


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

        self.model = Sequential(*self.layers)

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


class ForwardModelEncoderAeris(nn.Module):
    def __init__(self, encoder, action_dim, config, encoder_loss):
        super(ForwardModelEncoderAeris, self).__init__()

        self.action_dim = action_dim
        self.feature_dim = config.forward_model_kernels_count
        self.encoder = encoder
        self.encoder_loss = encoder_loss

        self.layers = [
            nn.Linear(self.feature_dim + self.action_dim, self.feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.Tanh()
        ]

        nn.init.xavier_uniform_(self.layers[0].weight)

        self.model = Sequential(*self.layers)

    def forward(self, state, action):
        features = self.encoder(state)
        x = torch.cat([features, action], dim=1)

        predicted_state = self.model(x)

        return predicted_state

    def error(self, state, action, next_state):
        with torch.no_grad():
            prediction = self(state, action)
            target = self.encoder(next_state)

            error = torch.mean(torch.pow(prediction - target, 2), dim=1).unsqueeze(1)

        return error

    def loss_function(self, state, action, next_state):
        target = self.encoder(next_state).detach()
        loss = nn.functional.mse_loss(self(state, action), target)
        if self.encoder_loss:
            loss += self.encoder.loss_function(state, next_state)
        return loss
