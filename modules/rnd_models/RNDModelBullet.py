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
            nn.ELU(),
            nn.Linear(fm_h[0], fm_h[1]),
            nn.ELU(),
            nn.Linear(fm_h[1], 1),
            nn.Tanh()
        )

        gain = np.sqrt(2)
        self._init(self.target_model[0], gain)
        self._init(self.target_model[2], gain)
        self._init(self.target_model[4], gain)

        for param in self.target_model.parameters():
            param.requires_grad = False

        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, fm_h[1]),
            nn.ELU(),
            nn.Linear(fm_h[1], 1),
            nn.Tanh()
        )

        gain = 1
        nn.init.xavier_uniform_(self.model[0].weight, gain)
        self.model[0].bias.data.zero_()
        nn.init.xavier_uniform_(self.model[2].weight, gain)
        self.model[2].bias.data.zero_()

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
            loss = nn.functional.mse_loss(self(state, action), self.encode(state, action).detach(), reduction='sum')
        else:
            loss = nn.functional.mse_loss(prediction, self.encode(state, action).detach(), reduction='sum')
        return loss

    def _init(self, layer, gain):
        nn.init.orthogonal_(layer.weight, gain)
        layer.bias.data.zero_()


class DOPSimpleModelBullet(nn.Module):
    def __init__(self, state_dim, action_dim, config, actor):
        super(DOPSimpleModelBullet, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.motivator = QRNDModelBullet(state_dim, action_dim, config)
        self.actor = actor

    def forward(self, state, action):
        predicted_code = self.motivator(state, action)
        return predicted_code

    def error(self, state, action):
        return self.motivator.error(state, action)

    def motivator_loss_function(self, state, action, prediction=None):
        return self.motivator.loss_function(state, action, prediction)

    def generator_loss_function(self, state):
        loss = -self.motivator.loss_function(state, self.actor(state))
        return loss

    def _init(self, layer, gain):
        nn.init.orthogonal_(layer.weight, gain)
        layer.bias.data.zero_()


class DOPModelBullet(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(DOPModelBullet, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.G = 16

        self.motivator = QRNDModelBullet(state_dim, action_dim, config)

        fm_h = [int(x) for x in config.fm_h.split(',')]

        self.generator = []

        for i in range(self.G):
            g = nn.Sequential(
                nn.Linear(state_dim + action_dim, fm_h[0]),
                nn.ReLU(),
                nn.Linear(fm_h[0], fm_h[1]),
                nn.ReLU(),
                nn.Linear(fm_h[1], action_dim),
                nn.Tanh()
            )

            self._init(g[0], np.sqrt(2))
            self._init(g[2], np.sqrt(2))
            self._init(g[4], np.sqrt(2))

            self.generator.append(g)

    def noise(self, state, action):
        noise = []
        motivation = []
        for i in range(self.G):
            x = torch.cat([state, action.detach()], dim=1)
            new_action = self.generator[i](x)
            noise.append(new_action)
            m = self.error(state, new_action)
            motivation.append(m)

        motivation = torch.stack(motivation)

        index = motivation.argmax(dim=0)

        return noise[index.item()], index

    def forward(self, state, action):
        predicted_code = self.motivator(state, action)
        return predicted_code

    def error(self, state, action):
        return self.motivator.error(state, action)

    def motivator_loss_function(self, state, action, prediction=None):
        return self.motivator.loss_function(state, action, prediction)

    def generator_loss_function(self, state, action, index):
        x = torch.cat([state, action], dim=1)
        loss = -self.motivator.loss_function(state, self.generator[index](x)) * 1e2
        return loss

    def _init(self, layer, gain):
        nn.init.orthogonal_(layer.weight, gain)
        layer.bias.data.zero_()
