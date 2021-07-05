import torch
import torch.nn as nn
import numpy as np

from modules import init


class RNDModelAtari(nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(RNDModelAtari, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim

        input_channels = 1
        input_height = self.input_shape[1]
        input_width = self.input_shape[2]
        self.feature_dim = 512
        self.state_average = torch.zeros((1, input_channels, input_height, input_width), device=config.device)

        fc_inputs_count = 64 * (input_width // 8) * (input_height // 8)

        self.target_model = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(fc_inputs_count, self.feature_dim)
        )

        init(self.target_model[0], np.sqrt(2))
        init(self.target_model[2], np.sqrt(2))
        init(self.target_model[4], np.sqrt(2))
        init(self.target_model[7], np.sqrt(2))

        for param in self.target_model.parameters():
            param.requires_grad = False

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(fc_inputs_count, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        init(self.model[0], np.sqrt(2))
        init(self.model[2], np.sqrt(2))
        init(self.model[4], np.sqrt(2))
        init(self.model[7], np.sqrt(2))
        init(self.model[9], np.sqrt(2))
        init(self.model[11], np.sqrt(2))

    def forward(self, state):
        x = state[:, 0, :, :].unsqueeze(1)
        predicted_code = self.model(x)
        return predicted_code

    def encode(self, state):
        x = state[:, 0, :, :].unsqueeze(1)
        return self.target_model(x)

    def error(self, state):
        x = state[:, 0, :, :].unsqueeze(1)
        with torch.no_grad():
            prediction = self(x)
            target = self.encode(state)
            error = torch.mean(torch.pow(prediction - target, 2), dim=1)

        return error

    def loss_function(self, state):
        loss = nn.functional.mse_loss(self(state), self.encode(state).detach(), reduction='none').sum(dim=1)
        mask = torch.empty_like(loss)
        mask = nn.init.uniform_(mask) < 0.25

        loss *= mask

        return loss.sum(dim=0) / mask.sum(dim=0)

    def update_state_average(self, state):
        pass
        # self.state_average = self.state_average * 0.99 + state[:, 0, :, :].unsqueeze(1) * 0.01
