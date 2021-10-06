import torch
import torch.nn as nn
import numpy as np

from modules import init_coupled_orthogonal, init_orthogonal


class RNDModelBitFlip(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(RNDModelBitFlip, self).__init__()

        self.state_average = torch.zeros((1, state_dim), device=config.device)

        self.action_dim = action_dim
        self.state_dim = state_dim

        self.target_model = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim * 4),
            nn.ELU(),
            nn.Linear(self.state_dim * 4, self.state_dim * 4),
            nn.ELU(),
            nn.Linear(self.state_dim * 4, self.state_dim * 4),
            nn.ELU(),
            nn.Linear(self.state_dim * 4, self.state_dim * 2),
        )

        for param in self.target_model.parameters():
            param.requires_grad = False

        self.model = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim * 4),
            nn.ELU(),
            nn.Linear(self.state_dim * 4, self.state_dim * 4),
            nn.ELU(),
            nn.Linear(self.state_dim * 4, self.state_dim * 4),
            nn.ELU(),
            nn.Linear(self.state_dim * 4, self.state_dim * 2),
            nn.ELU(),
            nn.Linear(self.state_dim * 2, self.state_dim * 2),
            nn.ELU(),
            nn.Linear(self.state_dim * 2, self.state_dim * 2),
        )

        init_coupled_orthogonal([self.target_model[0], self.model[0]], np.sqrt(2))
        init_coupled_orthogonal([self.target_model[2], self.model[2]], np.sqrt(2))
        init_coupled_orthogonal([self.target_model[4], self.model[4]], np.sqrt(2))
        init_coupled_orthogonal([self.target_model[6], self.model[6]], 1)
        init_orthogonal(self.model[8], 0.1)
        init_orthogonal(self.model[10], 0.01)

    def forward(self, state):
        x = state - self.state_average.expand(state.shape[0], *state.shape[1:])
        predicted_code = self.model(x)
        return predicted_code

    def encode(self, state):
        x = state - self.state_average.expand(state.shape[0], *state.shape[1:])
        return self.target_model(x)

    def error(self, state):
        with torch.no_grad():
            prediction = self(state)
            target = self.encode(state)
            error = torch.mean(torch.pow(prediction - target, 2), dim=1)

        return error

    def loss_function(self, state, predicted_state=None):
        if predicted_state is None:
            loss = nn.functional.mse_loss(self(state), self.encode(state).detach(), reduction='none').sum(dim=1)
        else:
            loss = nn.functional.mse_loss(predicted_state, self.encode(state).detach(), reduction='none').sum(dim=1)
        mask = torch.empty_like(loss)
        mask = nn.init.uniform_(mask) < 0.25

        loss *= mask

        return loss.sum(dim=0) / (mask.sum(dim=0) + 1e-8)

    def update_state_average(self, state):
        self.state_average = self.state_average * 0.999 + state * 0.001


class QRNDModelBitFlip(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(QRNDModelBitFlip, self).__init__()

        self.state_average = torch.zeros((1, state_dim), device=config.device)

        self.action_dim = action_dim
        self.state_dim = state_dim

        self.target_model = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, self.state_dim * 4),
            nn.ELU(),
            nn.Linear(self.state_dim * 4, self.state_dim * 4),
            nn.ELU(),
            nn.Linear(self.state_dim * 4, self.state_dim * 4),
            nn.ELU(),
            nn.Linear(self.state_dim * 4, self.state_dim * 2),
        )

        for param in self.target_model.parameters():
            param.requires_grad = False

        self.model = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, self.state_dim * 4),
            nn.ELU(),
            nn.Linear(self.state_dim * 4, self.state_dim * 4),
            nn.ELU(),
            nn.Linear(self.state_dim * 4, self.state_dim * 4),
            nn.ELU(),
            nn.Linear(self.state_dim * 4, self.state_dim * 2),
            nn.ELU(),
            nn.Linear(self.state_dim * 2, self.state_dim * 2),
            nn.ELU(),
            nn.Linear(self.state_dim * 2, self.state_dim * 2),
        )

        init_coupled_orthogonal([self.target_model[0], self.model[0]], np.sqrt(2))
        init_coupled_orthogonal([self.target_model[2], self.model[2]], np.sqrt(2))
        init_coupled_orthogonal([self.target_model[4], self.model[4]], np.sqrt(2))
        init_coupled_orthogonal([self.target_model[6], self.model[6]], 1)
        init_orthogonal(self.model[8], 0.1)
        init_orthogonal(self.model[10], 0.01)

    def forward(self, state, action):
        x = state - self.state_average.expand(state.shape[0], *state.shape[1:])
        x = torch.cat([x, action], dim=1)
        predicted_code = self.model(x)
        return predicted_code

    def encode(self, state, action):
        x = state - self.state_average.expand(state.shape[0], *state.shape[1:])
        x = torch.cat([x, action], dim=1)
        return self.target_model(x)

    def error(self, state, action):
        prediction = self(state, action)
        target = self.encode(state, action)
        error = torch.mean(torch.pow(prediction - target, 2), dim=1)

        return error

    def loss_function(self, state, action, prediction=None):
        if prediction is None:
            loss = nn.functional.mse_loss(self(state, action), self.encode(state, action).detach(), reduction='sum')
        else:
            loss = nn.functional.mse_loss(prediction, self.encode(state, action).detach(), reduction='sum')
        mask = torch.empty_like(loss)
        mask = nn.init.uniform_(mask) < 0.25

        loss *= mask

        return loss.sum(dim=0) / (mask.sum(dim=0) + 1e-8)

    def update_state_average(self, state):
        self.state_average = self.state_average * 0.999 + state * 0.001
