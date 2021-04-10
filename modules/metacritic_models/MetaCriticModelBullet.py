import torch
import torch.nn as nn
import numpy as np


class MetaCriticModelBullet(nn.Module):
    def __init__(self, forward_model, state_dim, action_dim, config):
        super(MetaCriticModelBullet, self).__init__()

        self.forward_model = forward_model
        self.state_dim = state_dim
        self.action_dim = action_dim

        mc_h = [int(x) for x in config.mc_h.split(',')]

        self.model = nn.Sequential(
            nn.Linear(state_dim, mc_h[0]),
            nn.ReLU(),
            nn.Linear(mc_h[0], mc_h[1]),
            nn.ReLU(),
            nn.Linear(mc_h[1], 1)
        )

        self._init(self.model[0], np.sqrt(2))
        self._init(self.model[2], np.sqrt(2))
        self._init(self.model[4], np.sqrt(2))

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
        error = torch.mean(torch.pow(predicted_state - next_state, 2), dim=1).unsqueeze(1)
        return error

    def loss_function(self, state, action, next_state):
        predicted_state, error_estimate = self(state, action)
        error = self._error(predicted_state, next_state).detach()
        loss = nn.functional.mse_loss(predicted_state, next_state) + nn.functional.mse_loss(error_estimate, error)
        return loss

    def _init(self, layer, gain):
        nn.init.orthogonal_(layer.weight, gain)
        layer.bias.data.zero_()
