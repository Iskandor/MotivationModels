import torch
import torch.nn as nn
import numpy as np


class RNDModelBullet(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(RNDModelBullet, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        fm_h = [int(x) for x in config.fm_h.split(',')]

        self.target_model = nn.Sequential(
            nn.Linear(state_dim, fm_h[0]),
            nn.ReLU(),
            nn.Linear(fm_h[0], state_dim)
        )

        self._init(self.target_model[0], np.sqrt(2))
        self._init(self.target_model[2], np.sqrt(2))

        for param in self.target_model.parameters():
            param.requires_grad = False

        self.model = nn.Sequential(
            nn.Linear(state_dim, fm_h[0]),
            nn.ReLU(),
            nn.Linear(fm_h[0], fm_h[1]),
            nn.ReLU(),
            nn.Linear(fm_h[1], state_dim)
        )

        self._init(self.model[0], np.sqrt(2))
        self._init(self.model[2], np.sqrt(2))
        self._init(self.model[4], np.sqrt(2))

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

    def loss_function(self, state, predicted_state=None):
        if predicted_state is None:
            loss = nn.functional.mse_loss(self(state), self.encode(state).detach())
        else:
            loss = nn.functional.mse_loss(predicted_state, self.encode(state).detach())
        return loss

    def _init(self, layer, gain):
        nn.init.orthogonal_(layer.weight, gain)
        layer.bias.data.zero_()


class QRNDModelBullet(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(QRNDModelBullet, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        fm_h = [int(x) for x in config.fm_h.split(',')]

        self.target_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, fm_h[0]),
            nn.ReLU(),
            nn.Linear(fm_h[0], state_dim + action_dim)
        )

        self._init(self.target_model[0], np.sqrt(2))
        self._init(self.target_model[2], np.sqrt(2))

        for param in self.target_model.parameters():
            param.requires_grad = False

        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, fm_h[0]),
            nn.ReLU(),
            nn.Linear(fm_h[0], fm_h[1]),
            nn.ReLU(),
            nn.Linear(fm_h[1], state_dim + action_dim)
        )

        self._init(self.model[0], np.sqrt(2))
        self._init(self.model[2], np.sqrt(2))
        self._init(self.model[4], np.sqrt(2))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        predicted_code = self.model(x)
        return predicted_code

    def encode(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.target_model(x)

    def error(self, state, action):
        with torch.no_grad():
            prediction = self(state, action)
            target = self.encode(state, action)
            error = torch.mean(torch.pow(prediction - target, 2), dim=1)

        return error

    def loss_function(self, state, action, prediction=None):
        if prediction is None:
            loss = nn.functional.mse_loss(self(state, action), self.encode(state, action).detach())
        else:
            loss = nn.functional.mse_loss(prediction, self.encode(state, action).detach())
        return loss

    def _init(self, layer, gain):
        nn.init.orthogonal_(layer.weight, gain)
        layer.bias.data.zero_()


class DOPModelBullet(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(DOPModelBullet, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.motivator = QRNDModelBullet(state_dim, action_dim, config)

        fm_h = [int(x) for x in config.fm_h.split(',')]

        self.generator = nn.Sequential(
            nn.Linear(state_dim, fm_h[0]),
            nn.ReLU(),
            nn.Linear(fm_h[0], fm_h[1]),
            nn.ReLU(),
            nn.Linear(fm_h[1], action_dim),
            nn.Tanh()
        )

        self._init(self.generator[0], np.sqrt(2))
        self._init(self.generator[2], np.sqrt(2))
        self._init(self.generator[4], np.sqrt(2))

    def noise(self, state):
        noise = self.generator(state)
        return noise

    def forward(self, state, action):
        predicted_code = self.motivator(state, action)
        return predicted_code

    def error(self, state, action):
        return self.motivator.error(state, action)

    def motivator_loss_function(self, state, action, prediction=None):
        return self.motivator.loss_function(state, action, prediction)

    def generator_loss_function(self, state, action):
        loss = -self.motivator.loss_function(state, action + self.generator(state))
        return loss

    def _init(self, layer, gain):
        nn.init.orthogonal_(layer.weight, gain)
        layer.bias.data.zero_()
