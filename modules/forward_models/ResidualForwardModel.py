import torch
from torch import nn
from torch.nn import *

from modules import ARCH

class ResidualForwardModel(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(ResidualForwardModel, self).__init__()

        self._model = Sequential(
            Linear(in_features=state_dim + action_dim, out_features=config.forward_model_h1, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h1, out_features=config.forward_model_h1, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h1, out_features=config.forward_model_h2, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h2, out_features=config.forward_model_h2, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h2, out_features=state_dim, bias=True)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], state.ndim - 1)
        predicted_state = self._model(x) + state
        return predicted_state

    def error(self, state, action, next_state):
        with torch.no_grad():
            dim = action.ndim - 1
            prediction = self(state, action)
            error = torch.mean(torch.pow(prediction.flatten(dim) - next_state.flatten(dim), 2), dim=dim).unsqueeze(dim)

        return error

    def loss_function(self, state, action, next_state):
        return nn.functional.mse_loss(self(state, action), next_state)
