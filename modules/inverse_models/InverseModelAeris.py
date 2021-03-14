import torch
from torch import nn
from torch.nn import *


class InverseModelAeris(nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(InverseModelAeris, self).__init__()

        self.channels = input_shape[0]
        self.width = input_shape[1]

        self.action_dim = action_dim

        fc_count = config.forward_model_kernels_count * self.width // 4

        self.feature_dim = config.forward_model_kernels_count

        channels = input_shape[0]

        self.layers_encoder = [
            nn.Conv1d(channels, config.forward_model_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_count, fc_count // 2),
            nn.ReLU(),
            nn.Linear(fc_count // 2, self.feature_dim),
            nn.Tanh()
        ]

        nn.init.xavier_uniform_(self.layers_encoder[0].weight)
        nn.init.xavier_uniform_(self.layers_encoder[3].weight)
        nn.init.xavier_uniform_(self.layers_encoder[5].weight)

        self.encoder = Sequential(*self.layers_encoder)

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