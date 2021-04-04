import torch
import torch.nn as nn
import numpy as np


class RNDModelAeris(nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(RNDModelAeris, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim

        self.channels = input_shape[0]

        self.target_model = nn.Sequential(
            nn.Conv1d(self.channels, config.forward_model_kernels_count, kernel_size=8, stride=4, padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(config.forward_model_kernels_count, config.forward_model_kernels_count * 2, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(config.forward_model_kernels_count * 2, config.forward_model_kernels_count * 2, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU()
        )

        self._init(self.target_model[0], np.sqrt(2))
        self._init(self.target_model[2], np.sqrt(2))
        self._init(self.target_model[4], np.sqrt(2))
        self._init(self.target_model[7], np.sqrt(2))

        for param in self.target_model.parameters():
            param.requires_grad = False

        self.model = nn.Sequential(
            nn.Conv1d(self.channels, config.forward_model_kernels_count, kernel_size=8, stride=4, padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(config.forward_model_kernels_count, config.forward_model_kernels_count * 2, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(config.forward_model_kernels_count * 2, config.forward_model_kernels_count * 2, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self._init(self.model[0], np.sqrt(2))
        self._init(self.model[2], np.sqrt(2))
        self._init(self.model[4], np.sqrt(2))
        self._init(self.model[7], np.sqrt(2))
        self._init(self.model[9], np.sqrt(2))
        self._init(self.model[11], np.sqrt(2))

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

    def loss_function(self, state):
        loss = nn.functional.mse_loss(self(state), self.encode(state).detach())
        return loss

    def _init(self, layer, gain):
        nn.init.orthogonal_(layer.weight, gain)
        layer.bias.data.zero_()