import torch
from torch import nn
from torch.nn import *


class InverseModelAeris(nn.Module):
    def __init__(self, encoder, action_dim, config):
        super(InverseModelAeris, self).__init__()

        self.action_dim = action_dim
        self.feature_dim = config.forward_model_kernels_count

        self.encoder = encoder

        self.layers = [
            nn.Linear(self.feature_dim * 2, self.action_dim),
            nn.Tanh()
        ]

        nn.init.xavier_uniform_(self.layers[0].weight)

        self.model = Sequential(*self.layers)

    def forward(self, state, next_state):
        features_state = self.encoder(state)
        features_next_state = self.encoder(next_state)
        x = torch.cat([features_state, features_next_state], dim=1)

        predicted_action = self.model(x)

        return predicted_action

    def error(self, state, action, next_state):
        with torch.no_grad():
            prediction = self(state, next_state)
            error = torch.mean(torch.pow(prediction - action, 2), dim=1).unsqueeze(1)

        return error

    def loss_function(self, state, action, next_state):
        loss = nn.functional.mse_loss(self(state, next_state), action)
        return loss