import torch
from torch import nn
from torch.nn import *


class ForwardModelAeris(nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(ForwardModelAeris, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim

        channels = input_shape[0]

        self.layers = [
            nn.Conv1d(channels + action_dim, config.forward_model_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(config.forward_model_kernels_count, config.forward_model_kernels_count, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(config.forward_model_kernels_count, config.forward_model_kernels_count, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(config.forward_model_kernels_count, config.forward_model_kernels_count, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(config.forward_model_kernels_count, config.forward_model_kernels_count, kernel_size=8, stride=4, padding=2, output_padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(config.forward_model_kernels_count, channels, kernel_size=3, stride=1, padding=1)
        ]

        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.xavier_uniform_(self.layers[2].weight)
        nn.init.xavier_uniform_(self.layers[4].weight)
        nn.init.xavier_uniform_(self.layers[6].weight)
        nn.init.xavier_uniform_(self.layers[8].weight)
        nn.init.uniform_(self.layers[10].weight, -0.3, 0.3)

        self.model = Sequential(*self.layers)

    def forward(self, state, action):
        a = action.unsqueeze(2).repeat(1, 1, state.shape[2])
        x = torch.cat([state, a], dim=1)

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


class ForwardModelEncoderAeris(nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(ForwardModelEncoderAeris, self).__init__()

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
            nn.Linear(self.feature_dim + self.action_dim, self.feature_dim)
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
        return loss
