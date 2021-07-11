import torch
import torch.nn as nn
import numpy as np


class RNDModelAeris(nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(RNDModelAeris, self).__init__()

        self.state_average = torch.zeros((1, input_shape[0], input_shape[1]))

        self.input_shape = input_shape
        self.action_dim = action_dim

        self.channels = input_shape[0]
        self.width = input_shape[1]

        fc_count = config.forward_model_kernels_count * self.width // 4

        self.target_model = nn.Sequential(
            nn.Conv1d(self.channels, config.forward_model_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(config.forward_model_kernels_count, config.forward_model_kernels_count * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(config.forward_model_kernels_count * 2, config.forward_model_kernels_count * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_count, 64)
        )

        self._init(self.target_model[0], np.sqrt(2))
        self._init(self.target_model[2], np.sqrt(2))
        self._init(self.target_model[4], np.sqrt(2))
        self._init(self.target_model[7], 1)

        for param in self.target_model.parameters():
            param.requires_grad = False

        self.model = nn.Sequential(
            nn.Conv1d(self.channels, config.forward_model_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(config.forward_model_kernels_count, config.forward_model_kernels_count * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(config.forward_model_kernels_count * 2, config.forward_model_kernels_count * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_count, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self._init(self.model[0], np.sqrt(2))
        self._init(self.model[2], np.sqrt(2))
        self._init(self.model[4], np.sqrt(2))
        self._init(self.model[7], 0.1)
        self._init(self.model[9], 0.1)
        self._init(self.model[11], 0.01)

    def forward(self, state):
        x = state - self.state_average
        predicted_code = self.model(x)
        return predicted_code

    def encode(self, state):
        x = state - self.state_average
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

    def _init(self, layer, gain):
        nn.init.orthogonal_(layer.weight, gain)
        layer.bias.data.zero_()


class QRNDModelAeris(nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(QRNDModelAeris, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim

        self.channels = input_shape[0]
        self.width = input_shape[1]

        self.state_average = torch.zeros((1, input_shape[0], input_shape[1]))

        fc_count = config.forward_model_kernels_count * self.width // 4

        self.target_model = nn.Sequential(
            nn.Conv1d(self.channels + action_dim, config.forward_model_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(config.forward_model_kernels_count, config.forward_model_kernels_count * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(config.forward_model_kernels_count * 2, config.forward_model_kernels_count * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_count, 64)
        )

        self._init(self.target_model[0], np.sqrt(2))
        self._init(self.target_model[2], np.sqrt(2))
        self._init(self.target_model[4], np.sqrt(2))
        self._init(self.target_model[7], 1)

        for param in self.target_model.parameters():
            param.requires_grad = False

        self.model = nn.Sequential(
            nn.Conv1d(self.channels + action_dim, config.forward_model_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(config.forward_model_kernels_count, config.forward_model_kernels_count * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(config.forward_model_kernels_count * 2, config.forward_model_kernels_count * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_count, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self._init(self.model[0], np.sqrt(2))
        self._init(self.model[2], np.sqrt(2))
        self._init(self.model[4], np.sqrt(2))
        self._init(self.model[7], 0.1)
        self._init(self.model[9], 0.1)
        self._init(self.model[11], 0.01)

    def forward(self, state, action):
        x = state - self.state_average
        a = action.unsqueeze(2).repeat(1, 1, state.shape[2])
        x = torch.cat([x, a], dim=1)
        predicted_code = self.model(x)
        return predicted_code

    def encode(self, state, action):
        x = state - self.state_average
        a = action.unsqueeze(2).repeat(1, 1, state.shape[2])
        x = torch.cat([x, a], dim=1)
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
        return loss

    def update_state_average(self, state):
        self.state_average = self.state_average * 0.999 + state * 0.001

    def _init(self, layer, gain):
        nn.init.orthogonal_(layer.weight, gain)
        layer.bias.data.zero_()


class DOPModelAeris(nn.Module):
    def __init__(self, state_dim, action_dim, config, features, actor, motivator):
        super(DOPModelAeris, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.motivator = motivator
        self.features = features
        self.actor = actor
        self.eta = config.motivation_eta

    def forward(self, state, action):
        predicted_code = self.motivator(state, action)
        return predicted_code

    def error(self, state, action):
        return self.motivator.error(state, action)

    def motivator_loss_function(self, state, action, prediction=None):
        return self.motivator.loss_function(state, action, prediction)

    def generator_loss_function(self, state):
        if self.features is not None:
            x = self.features(state)
        else:
            x = state
        action = self.actor(x)
        action = action.view(-1, self.action_dim)
        # prob = prob.view(-1, self.action_dim)
        state = state.unsqueeze(1).repeat(1, self.actor.head_count, 1, 1).view(-1, self.state_dim[0], self.state_dim[1])
        # loss = self.actor.log_prob(prob, action) * error.unsqueeze(-1) * self.eta
        loss = -self.error(state, action).unsqueeze(-1) * self.eta
        return loss.mean()
