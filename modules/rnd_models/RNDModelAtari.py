import torch
import torch.nn as nn
import numpy as np

from modules import init_orthogonal
from utils.RunningAverage import RunningStats


class RNDModelAtari(nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(RNDModelAtari, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim

        input_channels = 1
        input_height = self.input_shape[1]
        input_width = self.input_shape[2]
        self.feature_dim = 512
        self.state_average = RunningStats((4, input_height, input_width), config.device)

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

        init_orthogonal(self.target_model[0], np.sqrt(2))
        init_orthogonal(self.target_model[2], np.sqrt(2))
        init_orthogonal(self.target_model[4], np.sqrt(2))
        init_orthogonal(self.target_model[7], np.sqrt(2))

        for param in self.target_model.parameters():
            param.requires_grad = False

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(fc_inputs_count, self.feature_dim),
            nn.ELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        init_orthogonal(self.model[0], np.sqrt(2))
        init_orthogonal(self.model[2], np.sqrt(2))
        init_orthogonal(self.model[4], np.sqrt(2))
        init_orthogonal(self.model[7], np.sqrt(2))
        init_orthogonal(self.model[9], np.sqrt(2))
        init_orthogonal(self.model[11], np.sqrt(2))

    def prepare_input(self, state):
        x = state - self.state_average.mean
        return x[:, 0, :, :].unsqueeze(1)

    def forward(self, state):
        x = self.prepare_input(state)
        predicted_code = self.model(x)
        target_code = self.target_model(x)
        return predicted_code, target_code

    def error(self, state):
        with torch.no_grad():
            prediction, target = self(state)
            error = torch.sum(torch.pow(target - prediction, 2), dim=1).unsqueeze(-1) / 2

        return error

    def loss_function(self, state):
        prediction, target = self(state)
        # loss = nn.functional.mse_loss(self(state), self.encode(state).detach(), reduction='none').sum(dim=1)
        loss = torch.pow(target - prediction, 2)
        mask = torch.rand_like(loss) < 0.25
        loss *= mask

        return loss.sum() / mask.sum()

    def update_state_average(self, state):
        self.state_average.update(state)


class QRNDModelAtari(nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(QRNDModelAtari, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim

        input_channels = 1 + 1
        input_height = self.input_shape[1]
        input_width = self.input_shape[2]
        self.feature_dim = 512
        self.state_average = RunningStats((4, input_height, input_width), config.device)
        self.action_average = RunningStats((1, input_height, input_width), config.device)

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

        init_orthogonal(self.target_model[0], np.sqrt(2))
        init_orthogonal(self.target_model[2], np.sqrt(2))
        init_orthogonal(self.target_model[4], np.sqrt(2))
        init_orthogonal(self.target_model[7], np.sqrt(2))

        for param in self.target_model.parameters():
            param.requires_grad = False

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(fc_inputs_count, self.feature_dim),
            nn.ELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        init_orthogonal(self.model[0], np.sqrt(2))
        init_orthogonal(self.model[2], np.sqrt(2))
        init_orthogonal(self.model[4], np.sqrt(2))
        init_orthogonal(self.model[7], np.sqrt(2))
        init_orthogonal(self.model[9], np.sqrt(2))
        init_orthogonal(self.model[11], np.sqrt(2))

    def prepare_input(self, state, action):
        action = action.unsqueeze(2).unsqueeze(3).repeat(1, 1, state.shape[2], state.shape[3]).argmax(dim=1).unsqueeze(1) / self.action_dim
        action = action - self.action_average.mean
        state = state - self.state_average.mean
        x = torch.cat([state[:, 0, :, :].unsqueeze(1), action], dim=1)
        return x

    def forward(self, state, action):
        x = self.prepare_input(state, action)
        predicted_code = self.model(x)
        target_code = self.target_model(x)
        return predicted_code, target_code

    def error(self, state, action):
        with torch.no_grad():
            prediction, target = self(state, action.view(-1, self.action_dim))
            error = torch.sum(torch.pow(target - prediction, 2), dim=1).unsqueeze(-1) / 8
            # error = torch.mean(torch.pow(target - prediction, 2), dim=1).unsqueeze(-1)

        return error

    def loss_function(self, state, action):
        prediction, target = self(state, action)
        # loss = nn.functional.mse_loss(self(state), self.encode(state).detach(), reduction='none').sum(dim=1)
        loss = torch.pow(target - prediction, 2)
        mask = torch.rand_like(loss) < 0.5
        loss *= mask

        return loss.sum() / mask.sum()
        # return loss.mean()

    def update_state_average(self, state, action):
        action = action.unsqueeze(2).unsqueeze(3).repeat(1, 1, state.shape[2], state.shape[3]).argmax(dim=1).unsqueeze(1) / self.action_dim

        self.action_average.update(action)
        self.state_average.update(state)
