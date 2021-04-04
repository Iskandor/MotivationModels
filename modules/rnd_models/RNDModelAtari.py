import random

import torch
import torch.nn as nn
import numpy as np


class RNDModel(nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(RNDModel, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim

        self.target_model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512)
        )

        nn.init.orthogonal_(self.target_model[0].weights, np.sqrt(2))
        self.target_model[0].bias.data.zero_()
        nn.init.orthogonal_(self.target_model[2].weights, np.sqrt(2))
        self.target_model[2].bias.data.zero_()
        nn.init.orthogonal_(self.target_model[4].weights, np.sqrt(2))
        self.target_model[4].bias.data.zero_()
        nn.init.orthogonal_(self.target_model[7].weights, np.sqrt(2))
        self.target_model[7].bias.data.zero_()

        for param in self.target_model.parameters():
            param.requires_grad = False

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        nn.init.orthogonal_(self.model[0].weights, np.sqrt(2))
        self.model[0].bias.data.zero_()
        nn.init.orthogonal_(self.model[2].weights, np.sqrt(2))
        self.model[2].bias.data.zero_()
        nn.init.orthogonal_(self.model[4].weights, np.sqrt(2))
        self.model[4].bias.data.zero_()
        nn.init.orthogonal_(self.model[7].weights, np.sqrt(2))
        self.model[7].bias.data.zero_()
        nn.init.orthogonal_(self.model[9].weights, np.sqrt(2))
        self.model[9].bias.data.zero_()
        nn.init.orthogonal_(self.model[9].weights, np.sqrt(2))
        self.model[9].bias.data.zero_()

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