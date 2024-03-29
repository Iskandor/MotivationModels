import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import distance

from modules import init_orthogonal, init_coupled_orthogonal, init_xavier_uniform
from utils import one_hot_code
from utils.RunningAverage import RunningStats


class RNDModelAeris(nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(RNDModelAeris, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim

        self.channels = input_shape[0]
        self.width = input_shape[1]

        self.state_stats = RunningStats((self.channels, self.width), config.device)

        fc_count = config.forward_model_kernels_count * self.width // 4
        hidden_count = config.rnd_output

        self.target_model = nn.Sequential(
            nn.Conv1d(self.channels, config.forward_model_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ELU(),
            nn.Conv1d(config.forward_model_kernels_count, config.forward_model_kernels_count * 2, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv1d(config.forward_model_kernels_count * 2, config.forward_model_kernels_count * 2, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(fc_count, hidden_count)
        )

        for param in self.target_model.parameters():
            param.requires_grad = False

        self.model = nn.Sequential(
            nn.Conv1d(self.channels, config.forward_model_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ELU(),
            nn.Conv1d(config.forward_model_kernels_count, config.forward_model_kernels_count * 2, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv1d(config.forward_model_kernels_count * 2, config.forward_model_kernels_count * 2, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(fc_count, hidden_count),
            nn.ELU(),
            nn.Linear(hidden_count, hidden_count),
            nn.ELU(),
            nn.Linear(hidden_count, hidden_count)
        )

        init_coupled_orthogonal([self.target_model[0], self.model[0]], np.sqrt(2))
        init_coupled_orthogonal([self.target_model[2], self.model[2]], np.sqrt(2))
        init_coupled_orthogonal([self.target_model[4], self.model[4]], np.sqrt(2))
        init_coupled_orthogonal([self.target_model[7], self.model[7]], 1)
        init_orthogonal(self.model[9], 0.1)
        init_orthogonal(self.model[11], 0.01)

    def forward(self, state):
        x = state - self.state_stats.mean.expand(state.shape[0], *state.shape[1:])
        predicted_code = self.model(x)
        return predicted_code

    def encode(self, state):
        x = state - self.state_stats.mean.expand(state.shape[0], *state.shape[1:])
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
        self.state_stats.update(state)


class QRNDModelAeris(nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(QRNDModelAeris, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim

        self.channels = input_shape[0] + action_dim
        self.width = input_shape[1]

        self.state_stats = RunningStats((self.channels, self.width), config.device)

        fc_count = config.forward_model_kernels_count * self.width // 4
        hidden_count = config.rnd_output

        self.target_model_features = nn.Sequential(
            nn.Conv1d(self.channels, config.forward_model_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(config.forward_model_kernels_count, config.forward_model_kernels_count * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(config.forward_model_kernels_count * 2, config.forward_model_kernels_count * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        for param in self.target_model_features.parameters():
            param.requires_grad = False

        self.target_model = nn.Sequential(
            nn.Linear(fc_count, hidden_count)
        )

        for param in self.target_model.parameters():
            param.requires_grad = False

        self.model_features = nn.Sequential(
            nn.Conv1d(self.channels, config.forward_model_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(config.forward_model_kernels_count, config.forward_model_kernels_count * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(config.forward_model_kernels_count * 2, config.forward_model_kernels_count * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.model = nn.Sequential(
            nn.Linear(fc_count, hidden_count),
            nn.ReLU(),
            nn.Linear(hidden_count, hidden_count),
            nn.ReLU(),
            nn.Linear(hidden_count, hidden_count)
        )

        init_coupled_orthogonal([self.target_model_features[0], self.model_features[0]], np.sqrt(2))
        init_coupled_orthogonal([self.target_model_features[2], self.model_features[2]], np.sqrt(2))
        init_coupled_orthogonal([self.target_model_features[4], self.model_features[4]], np.sqrt(2))
        init_coupled_orthogonal([self.target_model[0], self.model[0]], 0.1)
        init_orthogonal(self.model[2], 0.1)
        init_orthogonal(self.model[4], 0.01)

    def prepare_input(self, state, action):
        a = action.unsqueeze(2).repeat(1, 1, state.shape[2])
        x = torch.cat([state, a], dim=1)
        mean = self.state_stats.mean.expand(x.shape[0], *x.shape[1:])
        x = (x - mean)
        return x

    def forward(self, state, action):
        x = self.prepare_input(state, action)
        x = self.model_features(x)
        predicted_code = self.model(x)
        return predicted_code

    def encode(self, state, action):
        x = self.prepare_input(state, action)
        x = self.target_model_features(x)
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

    def update_state_average(self, state, action):
        a = action.unsqueeze(2).repeat(1, 1, state.shape[2])
        x = torch.cat([state, a], dim=1)
        self.state_stats.update(x)


class VanillaQRNDModelAeris(nn.Module):
    def __init__(self, input_shape, action_dim, config, init='orto'):
        super(VanillaQRNDModelAeris, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim

        self.channels = input_shape[0] + action_dim
        self.width = input_shape[1]

        self.state_stats = RunningStats((self.channels, self.width), config.device)

        fc_count = config.forward_model_kernels_count * self.width // 4
        hidden_count = config.rnd_output

        self.target_model = nn.Sequential(
            nn.Conv1d(self.channels, config.forward_model_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(config.forward_model_kernels_count, config.forward_model_kernels_count * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(config.forward_model_kernels_count * 2, config.forward_model_kernels_count * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_count, hidden_count)
        )

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
            nn.Linear(fc_count, hidden_count),
            nn.ReLU(),
            nn.Linear(hidden_count, hidden_count),
            nn.ReLU(),
            nn.Linear(hidden_count, hidden_count)
        )

        if init == 'orto':
            self._orto_init()
        if init == 'corto':
            self._coupled_orto_init()

    def _orto_init(self):
        init_orthogonal(self.target_model[0], np.sqrt(2))
        init_orthogonal(self.model[0], np.sqrt(2))
        init_orthogonal(self.target_model[2], np.sqrt(2))
        init_orthogonal(self.model[2], np.sqrt(2))
        init_orthogonal(self.target_model[4], np.sqrt(2))
        init_orthogonal(self.model[4], np.sqrt(2))
        init_orthogonal(self.target_model[7], 0.1)
        init_orthogonal(self.model[7], 0.1)
        init_orthogonal(self.model[9], 0.1)
        init_orthogonal(self.model[11], 0.01)

    def _coupled_orto_init(self):
        init_coupled_orthogonal([self.target_model[0], self.model[0]], np.sqrt(2))
        init_coupled_orthogonal([self.target_model[2], self.model[2]], np.sqrt(2))
        init_coupled_orthogonal([self.target_model[4], self.model[4]], np.sqrt(2))
        init_orthogonal(self.target_model[7], 0.1)
        init_orthogonal(self.model[7], 0.1)
        init_orthogonal(self.model[9], 0.1)
        init_orthogonal(self.model[11], 0.01)

    def prepare_input(self, state, action):
        a = action.unsqueeze(2).repeat(1, 1, state.shape[2])
        x = torch.cat([state, a], dim=1)
        mean = self.state_stats.mean.expand(x.shape[0], *x.shape[1:])
        x = (x - mean)
        return x

    def forward(self, state, action):
        x = self.prepare_input(state, action)
        predicted_code = self.model(x)
        return predicted_code

    def encode(self, state, action):
        x = self.prepare_input(state, action)
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

    def update_state_average(self, state, action):
        a = action.unsqueeze(2).repeat(1, 1, state.shape[2])
        x = torch.cat([state, a], dim=1)
        self.state_stats.update(x)


class QRNDModelAerisFC(nn.Module):
    def __init__(self, input_shape, action_dim, config, init='orto'):
        super(QRNDModelAerisFC, self).__init__()

        self.state_dim = input_shape[0] * input_shape[1]
        self.action_dim = action_dim

        self.state_stats = RunningStats((1, self.state_dim + (self.action_dim * self.state_dim)), config.device)

        hidden_count = config.rnd_output

        self.target_model = nn.Sequential(
            nn.Linear(self.state_dim + (self.action_dim * self.state_dim), self.state_dim * 2),
            nn.ELU(),
            nn.Linear(self.state_dim * 2, self.state_dim * 2),
            nn.ELU(),
            nn.Linear(self.state_dim * 2, self.state_dim * 2),
            nn.ELU(),
            nn.Linear(self.state_dim * 2, hidden_count)
        )

        for param in self.target_model.parameters():
            param.requires_grad = False

        self.model = nn.Sequential(
            nn.Linear(self.state_dim + (self.action_dim * self.state_dim), self.state_dim * 2),
            nn.ELU(),
            nn.Linear(self.state_dim * 2, self.state_dim * 2),
            nn.ELU(),
            nn.Linear(self.state_dim * 2, self.state_dim * 2),
            nn.ELU(),
            nn.Linear(self.state_dim * 2, hidden_count),
            nn.ReLU(),
            nn.Linear(hidden_count, hidden_count),
            nn.ReLU(),
            nn.Linear(hidden_count, hidden_count)
        )

        if init == 'xavier':
            self._xavier_init()
        if init == 'orto':
            self._orto_init()
        if init == 'corto':
            self._coupled_orto_init()

    def _orto_init(self):
        init_orthogonal(self.target_model[0], np.sqrt(2))
        init_orthogonal(self.model[0], np.sqrt(2))
        init_orthogonal(self.target_model[2], np.sqrt(2))
        init_orthogonal(self.model[2], np.sqrt(2))
        init_orthogonal(self.target_model[4], np.sqrt(2))
        init_orthogonal(self.model[4], np.sqrt(2))
        init_orthogonal(self.target_model[6], 0.1)
        init_orthogonal(self.model[6], 0.1)
        init_orthogonal(self.model[8], 0.1)
        init_orthogonal(self.model[10], 0.01)

    def _xavier_init(self):
        init_xavier_uniform(self.target_model[0], np.sqrt(2))
        init_xavier_uniform(self.model[0], np.sqrt(2))
        init_xavier_uniform(self.target_model[2], np.sqrt(2))
        init_xavier_uniform(self.model[2], np.sqrt(2))
        init_xavier_uniform(self.target_model[4], np.sqrt(2))
        init_xavier_uniform(self.model[4], np.sqrt(2))
        init_xavier_uniform(self.target_model[6], 0.1)
        init_xavier_uniform(self.model[6], 0.1)
        init_xavier_uniform(self.model[8], 0.1)
        init_xavier_uniform(self.model[10], 0.01)

    def _coupled_orto_init(self):
        init_coupled_orthogonal([self.target_model[0], self.model[0]], np.sqrt(2))
        init_coupled_orthogonal([self.target_model[2], self.model[2]], np.sqrt(2))
        init_coupled_orthogonal([self.target_model[4], self.model[4]], np.sqrt(2))
        init_orthogonal(self.target_model[6], 0.1)
        init_orthogonal(self.model[6], 0.1)
        init_orthogonal(self.model[8], 0.1)
        init_orthogonal(self.model[10], 0.01)

    def prepare_input(self, state, action):
        x = state.view(state.shape[0], -1)
        a = action.unsqueeze(2).repeat(1, 1, x.shape[1]).view(x.shape[0], -1)
        x = torch.cat([x, a], dim=1)
        mean = self.state_stats.mean.expand(x.shape[0], *x.shape[1:])
        std = self.state_stats.std.expand(x.shape[0], *x.shape[1:])
        x = (x - mean) / (std + 1e-8)
        # x = (x - mean)

        return x

    def forward(self, state, action):
        x = self.prepare_input(state, action)
        predicted_code = self.model(x)
        return predicted_code

    def encode(self, state, action):
        x = self.prepare_input(state, action)
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
        # mask = torch.empty_like(loss)
        # mask = nn.init.uniform_(mask) < 0.25
        #
        # loss *= mask

        # return loss.sum(dim=0) / (mask.sum(dim=0) + 1e-8)

        return loss.sum(dim=0)

    def update_state_average(self, state, action):
        x = state.view(state.shape[0], -1)
        a = action.unsqueeze(2).repeat(1, 1, x.shape[1]).view(x.shape[0], -1)
        x = torch.cat([x, a], dim=1)
        self.state_stats.update(x)


class VanillaDOPModelAeris(nn.Module):
    def __init__(self, head_count, state_dim, action_dim, config, features, actor, motivator):
        super(VanillaDOPModelAeris, self).__init__()

        self.head_count = head_count
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.motivator = motivator
        self.features = features
        self.actor = actor
        self.eta = config.motivation_eta
        self.zeta = config.motivation_zeta

        self.error_stats = RunningStats(1, config.device)

    def forward(self, state, action):
        predicted_code = self.motivator(state, action)
        return predicted_code

    def error(self, state):
        action = self.actor(state).view(-1, self.action_dim)
        state = state.unsqueeze(1).repeat(1, self.head_count, 1, 1).view(-1, self.state_dim[0], self.state_dim[1])
        std = self.error_stats.std

        # return self.motivator.error(state, action) / (std + 1e-8)
        return self.motivator.error(state, action)

    def motivator_loss_function(self, state, action, prediction=None):
        return self.motivator.loss_function(state, action, prediction)

    def generator_loss_function(self, state):
        loss = -self.error(state).mean()
        reg_loss = self.regularization_term(state).mean() * self.zeta

        return (loss + reg_loss) * self.eta

    def update_error_average(self, error):
        self.error_stats.update(error)

    def regularization_term(self, state):
        action = self.actor(state)
        # y = torch.nn.functional.cosine_similarity(action[..., None, :, :], action[..., :, None, :].detach(), dim=-1) - 1
        y = -torch.cdist(action, action.detach(), p=1)

        return y


class DOPModelAeris(nn.Module):
    def __init__(self, head_count, state_dim, action_dim, config, features, actor, motivator):
        super(DOPModelAeris, self).__init__()

        self.head_count = head_count
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.motivator = motivator
        self.features = features
        self.actor = actor
        self.eta = config.motivation_eta
        self.zeta = config.motivation_zeta
        self.mask = None

        self.log_loss = []
        self.log_regterm = []

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

        # prob = prob.view(-1, self.action_dim)
        # loss = self.actor.log_prob(prob, action) * error.unsqueeze(-1) * self.eta

        action = self.actor(x).view(-1, self.action_dim)
        state = state.unsqueeze(1).repeat(1, self.head_count, 1, 1).view(-1, self.state_dim[0], self.state_dim[1])
        loss = -self.error(state, action).mean()

        regularization_term = self.regularization_term(action.view(-1, self.head_count, self.action_dim)) * self.zeta

        self.log_loss.append(-loss.item() * self.eta)
        self.log_regterm.append(-regularization_term.item() * self.eta)

        return (loss + regularization_term) * self.eta

    def regularization_term(self, action):
        # mask = torch.empty(action.shape[0], 1, device=action.device)
        # mask = nn.init.uniform_(mask) < self.zeta
        # action *= mask

        repulsive_term = torch.cdist(action, action.detach())

        return repulsive_term.mean()


class DOPV2ModelAeris(nn.Module):
    def __init__(self, head_count, state_dim, action_dim, config, features, actor, motivator, arbiter):
        super(DOPV2ModelAeris, self).__init__()

        self.head_count = head_count
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.motivator = motivator
        self.features = features
        self.actor = actor
        self.arbiter = arbiter
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
        state = state.unsqueeze(1).repeat(1, self.head_count, 1, 1).view(-1, self.state_dim[0], self.state_dim[1])
        error = self.error(state, action)

        index = self.arbiter(x)
        index_target = one_hot_code(torch.argmax(error.view(-1, self.head_count, 1).detach(), dim=1), self.head_count)

        loss_arbiter = nn.functional.mse_loss(index, index_target, reduction='mean')
        loss_generator = -error.unsqueeze(-1).mean()
        return (loss_generator * self.eta) + loss_arbiter


class DOPV3ModelAeris(nn.Module):
    def __init__(self, head_count, state_dim, action_dim, config, features, actor, critic, motivator):
        super(DOPV3ModelAeris, self).__init__()

        self.head_count = head_count
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.motivator = motivator
        self.features = features
        self.actor = actor
        self.critic = critic
        self.eta = config.motivation_eta
        self.zeta = config.motivation_zeta

        self.log_loss = []

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
        action = self.actor(x).view(-1, self.action_dim)
        state = state.unsqueeze(1).repeat(1, self.head_count, 1, 1).view(-1, self.state_dim[0], self.state_dim[1])

        _, value_int = self.critic(state, action)
        actor_loss = -value_int.mean()

        self.log_loss.append(-actor_loss.item())

        return actor_loss * self.eta


class DOPV2QModelAeris(nn.Module):
    def __init__(self, state_dim, action_dim, config, features, actor, motivator, arbiter):
        super(DOPV2QModelAeris, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.motivator = motivator
        self.features = features
        self.actor = actor
        self.arbiter = arbiter
        self.eta = config.motivation_eta

    def forward(self, state, action):
        predicted_code = self.motivator(state, action)
        return predicted_code

    def error(self, state, action):
        return self.motivator.error(state, action)

    def motivator_loss_function(self, state, action, prediction=None):
        return self.motivator.loss_function(state, action, prediction)

    def generator_loss_function(self, state, next_state, mask, gamma):
        if self.features is not None:
            x = self.features(state)
        else:
            x = state

        action = self.actor(x).view(-1, self.action_dim)
        s = x.unsqueeze(1).repeat(1, self.actor.head_count, 1, 1).view(-1, self.state_dim[0], self.state_dim[1])
        error = self.error(s, action)

        error_max = error.view(-1, self.actor.head_count).max(dim=1)
        index = error_max.indices.unsqueeze(-1)
        intrinsic_reward = error_max.values.unsqueeze(-1)

        current_q_value = self.arbiter(state).gather(dim=1, index=index)
        max_next_q_value = self.arbiter(next_state).max(dim=1).values.unsqueeze(-1)
        expected_q_values = intrinsic_reward + mask * (gamma * max_next_q_value)

        loss_arbiter = torch.nn.functional.mse_loss(current_q_value, expected_q_values.detach(), reduction='sum')

        loss_generator = -error.unsqueeze(-1).mean()
        return (loss_generator * self.eta) + loss_arbiter
