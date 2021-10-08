import torch
import torch.nn as nn
import numpy as np

from modules import init_coupled_orthogonal, init_orthogonal
from utils import one_hot_code


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
        init_coupled_orthogonal([self.target_model[6], self.model[6]], 0.1)
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


class DOPModelBitFlip(nn.Module):
    def __init__(self, head_count, state_dim, action_dim, config, features, actor, motivator):
        super(DOPModelBitFlip, self).__init__()

        self.head_count = head_count
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.motivator = motivator
        self.features = features
        self.actor = actor
        self.eta = config.motivation_eta
        self.zeta = config.motivation_zeta
        self.mask = None

    def forward(self, state, action):
        predicted_code = self.motivator(state, action)
        return predicted_code

    def error(self, state, action):
        state = state.unsqueeze(1).repeat(1, self.head_count, 1).view(-1, self.state_dim)
        action = action.view(-1, 1)
        action_code = one_hot_code(action, self.action_dim).view(-1, self.action_dim)
        return self.motivator.error(state, action_code)

    def motivator_loss_function(self, state, action, prediction=None):
        return self.motivator.loss_function(state, action, prediction)

    def generator_loss_function(self, state):
        if self.features is not None:
            x = self.features(state)
        else:
            x = state

        action, prob = self.actor(x)
        error = self.error(state, action)

        action = one_hot_code(action.view(-1, 1), self.action_dim).view(-1, self.action_dim)
        prob = prob.view(-1, self.action_dim)

        loss = self.actor.log_prob(prob, action) * error.unsqueeze(-1)
        loss = -loss.sum()

        return loss * self.eta
