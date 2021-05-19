import torch
import torch.nn as nn
import numpy as np


class RNDModelBullet(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(RNDModelBullet, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        fm_h = [int(x) for x in config.fm_h.split(',')]

        self.target_model = nn.Sequential(
            nn.Linear(state_dim, fm_h[0]),
            nn.ReLU(),
            nn.Linear(fm_h[0], state_dim)
        )

        self._init(self.target_model[0], np.sqrt(2))
        self._init(self.target_model[2], np.sqrt(2))

        for param in self.target_model.parameters():
            param.requires_grad = False

        self.model = nn.Sequential(
            nn.Linear(state_dim, fm_h[0]),
            nn.ReLU(),
            nn.Linear(fm_h[0], fm_h[1]),
            nn.ReLU(),
            nn.Linear(fm_h[1], state_dim)
        )

        self._init(self.model[0], np.sqrt(2))
        self._init(self.model[2], np.sqrt(2))
        self._init(self.model[4], np.sqrt(2))

    def forward(self, state):
        predicted_code = self.model(state)
        return predicted_code

    def encode(self, state):
        return self.target_model(state)

    def error(self, state):
        with torch.no_grad():
            prediction = self(state)
            target = self.encode(state)
            error = torch.mean(torch.pow(prediction - target, 2), dim=1)

        return error

    def loss_function(self, state, predicted_state=None):
        if predicted_state is None:
            loss = nn.functional.mse_loss(self(state), self.encode(state).detach())
        else:
            loss = nn.functional.mse_loss(predicted_state, self.encode(state).detach())
        return loss

    def _init(self, layer, gain):
        nn.init.orthogonal_(layer.weight, gain)
        layer.bias.data.zero_()
