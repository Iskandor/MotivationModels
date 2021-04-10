import torch
from torch import nn
from torch.nn import *
import numpy as np


class ForwardModelBullet(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(ForwardModelBullet, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        fm_h = [int(x) for x in config.fm_h.split(',')]

        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, fm_h[0]),
            nn.ReLU(),
            nn.Linear(fm_h[0], fm_h[1]),
            nn.ReLU(),
            nn.Linear(fm_h[1], state_dim))

        self._init(self.model[0], np.sqrt(2))
        self._init(self.model[2], np.sqrt(2))
        self._init(self.model[4], np.sqrt(2))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)

        predicted_state = self.model(x)

        return predicted_state

    def error(self, state, action, next_state):
        with torch.no_grad():
            prediction = self(state, action)
            error = torch.mean(torch.pow(prediction.view(prediction.shape[0], -1) - next_state.view(next_state.shape[0], -1), 2), dim=1).unsqueeze(1)

        return error

    def loss_function(self, state, action, next_state):
        loss = nn.functional.mse_loss(self(state, action), next_state)
        return loss

    def _init(self, layer, gain):
        nn.init.orthogonal_(layer.weight, gain)
        layer.bias.data.zero_()
