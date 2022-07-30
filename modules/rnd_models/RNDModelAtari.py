from math import sqrt

import torch
import torch.nn as nn
import numpy as np

from analytic.CNDAnalytic import CNDAnalytic
from analytic.RNDAnalytic import RNDAnalytic
from modules import init_orthogonal
from modules.encoders.EncoderAtari import ST_DIMEncoderAtari
from utils.RunningAverage import RunningStatsSimple


class RNDModelAtari(nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(RNDModelAtari, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim

        input_channels = 1
        input_height = self.input_shape[1]
        input_width = self.input_shape[2]
        self.feature_dim = 512
        self.state_average = RunningStatsSimple((4, input_height, input_width), config.device)

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
        loss = loss.sum() / mask.sum()

        analytic = RNDAnalytic()
        analytic.update(loss_prediction=loss.unsqueeze(-1).detach())

        return loss

    def update_state_average(self, state):
        self.state_average.update(state)


class CNDModelAtari(nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(CNDModelAtari, self).__init__()

        self.config = config
        self.action_dim = action_dim

        input_channels = 1
        # input_channels = input_shape[0]
        input_height = input_shape[1]
        input_width = input_shape[2]
        self.input_shape = (input_channels, input_height, input_width)
        self.feature_dim = 512

        fc_inputs_count = 128 * (input_width // 8) * (input_height // 8)

        self.state_average = RunningStatsSimple((4, input_height, input_width), config.device)

        self.target_model = ST_DIMEncoderAtari(self.input_shape, self.feature_dim, config)

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_inputs_count, self.feature_dim),
            nn.ELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        gain = sqrt(2)
        init_orthogonal(self.model[0], gain)
        init_orthogonal(self.model[2], gain)
        init_orthogonal(self.model[4], gain)
        init_orthogonal(self.model[6], gain)
        init_orthogonal(self.model[9], gain)
        init_orthogonal(self.model[11], gain)
        init_orthogonal(self.model[13], gain)

    def preprocess(self, state):
        if self.config.cnd_preprocess == 0:
            x = state
        if self.config.cnd_preprocess == 1:
            x = state - self.state_average.mean
        if self.config.cnd_preprocess == 2:
            x = ((state - self.state_average.mean) / self.state_average.std).clip(-1., 1.)

        return x[:, 0, :, :].unsqueeze(1)
        # return x

    def forward(self, state, fmaps=False):
        s = self.preprocess(state)

        f5 = self.model[:6](s)
        predicted_code = self.model[6:](f5)

        if fmaps:
            target_code = self.target_model(s, fmaps)

            return {
                'predicted_f5': f5.permute(0, 2, 3, 1),
                'predicted_code': predicted_code,
                'target_f5': target_code['f5'],
                'target_code': target_code['out']
            }
        else:
            target_code = self.target_model(s).detach()

        return predicted_code, target_code

    def error(self, state):
        with torch.no_grad():
            prediction, target = self(state)

            # if self.config.cnd_error_k == 2:
            #     error = torch.mean(torch.pow(target - prediction, 2), dim=1, keepdim=True)
            # if self.config.cnd_error_k == 1:
            #     error = torch.mean(torch.abs(target - prediction), dim=1, keepdim=True)

            error = self.k_distance(self.config.cnd_error_k, prediction, target, reduction='mean')

        return error

    def loss_function_crossentropy(self, state, next_state):
        out = self(state, fmaps=True)
        prediction_f5, prediction, target_f5, target = out['predicted_f5'], out['predicted_code'], out['target_f5'], out['target_code']

        loss_prediction = nn.functional.mse_loss(prediction, target.detach(), reduction='sum') + nn.functional.mse_loss(prediction_f5, target_f5.detach(), reduction='sum')
        # + nn.functional.mse_loss(prediction.detach(), target, reduction='mean')

        loss_target, loss_target_reg, loss_target_norm = self.target_model.loss_function_crossentropy(self.preprocess(state), self.preprocess(next_state))

        beta1 = 0
        beta2 = self.config.cnd_loss_target_reg

        analytic = CNDAnalytic()
        analytic.update(loss_prediction=loss_prediction.unsqueeze(-1).detach(), loss_target=loss_target.unsqueeze(-1).detach(), loss_reg=loss_target_reg.unsqueeze(-1).detach() * beta1, loss_target_norm=loss_target_norm.detach() * beta2)

        return loss_prediction * self.config.cnd_loss_pred + (loss_target + loss_target_reg * beta1 + loss_target_norm * beta2) * self.config.cnd_loss_target

    def loss_function_cdist(self, state, next_state):
        prediction, target = self(state)
        loss_prediction = nn.functional.mse_loss(prediction, target)
        loss_target = self.target_model.loss_function_cdist(self.preprocess(state), self.preprocess(next_state))

        analytic = CNDAnalytic()
        analytic.update(loss_prediction=loss_prediction.unsqueeze(-1).detach(), loss_target=loss_target.unsqueeze(-1).detach(), loss_reg=torch.zeros(1), loss_target_norm=torch.zeros(1))

        return loss_prediction * self.config.cnd_loss_pred + loss_target * self.config.cnd_loss_target

    def loss_function(self, state, next_state):
        return self.loss_function_crossentropy(state, next_state)

    @staticmethod
    def k_distance(k, prediction, target, reduction='sum'):
        ret = torch.abs(target - prediction) + 1e-8
        if reduction == 'sum':
            ret = ret.pow(k).sum(dim=1, keepdim=True)
        if reduction == 'mean':
            ret = ret.pow(k).mean(dim=1, keepdim=True)

        return ret

    def update_state_average(self, state):
        self.state_average.update(state)


class FEDRefModelAtari(nn.Module):
    def __init__(self, input_shape, action_dim, target_model, config):
        super(FEDRefModelAtari, self).__init__()

        self.config = config
        self.action_dim = action_dim

        input_channels = input_shape[0]
        input_height = input_shape[1]
        input_width = input_shape[2]
        self.input_shape = (input_channels, input_height, input_width)
        self.feature_dim = 512

        fc_inputs_count = 128 * (input_width // 8) * (input_height // 8)

        self.state_average = RunningStatsSimple((4, input_height, input_width), config.device)

        self.target_model = target_model

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_inputs_count, self.feature_dim),
            nn.ELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        gain = 1
        init_orthogonal(self.model[0], gain)
        init_orthogonal(self.model[2], gain)
        init_orthogonal(self.model[4], gain)
        init_orthogonal(self.model[6], gain)
        init_orthogonal(self.model[9], gain)
        init_orthogonal(self.model[11], gain)
        init_orthogonal(self.model[13], gain)

    def forward(self, state):
        predicted_code = self.model(state)
        target_code = self.target_model(state).detach()

        return predicted_code, target_code

    def error(self, state):
        with torch.no_grad():
            prediction, target = self(state)

            error = self.k_distance(2, prediction, target, reduction='mean')

        return error

    def loss_function(self, state):
        prediction, target = self(state)

        loss_prediction = nn.functional.mse_loss(prediction, target.detach(), reduction='sum')

        analytic = RNDAnalytic()
        analytic.update(loss_prediction=loss_prediction.unsqueeze(-1).detach())

        return loss_prediction

    @staticmethod
    def k_distance(k, prediction, target, reduction='sum'):
        ret = torch.abs(target - prediction) + 1e-8
        if reduction == 'sum':
            ret = ret.pow(k).sum(dim=1, keepdim=True)
        if reduction == 'mean':
            ret = ret.pow(k).mean(dim=1, keepdim=True)

        return ret

    def update_state_average(self, state):
        self.state_average.update(state)


class QRNDModelAtari(nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(QRNDModelAtari, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim

        input_channels = 1
        input_height = self.input_shape[1]
        input_width = self.input_shape[2]
        self.feature_dim = 512
        self.state_average = RunningStatsSimple((4, input_height, input_width), config.device)
        self.action_average = RunningStatsSimple(action_dim, config.device)

        fc_inputs_count = 64 * (input_width // 8) * (input_height // 8)

        self.target_model_state = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(fc_inputs_count, self.feature_dim),
            nn.ELU()
        )

        init_orthogonal(self.target_model_state[0], np.sqrt(2))
        init_orthogonal(self.target_model_state[2], np.sqrt(2))
        init_orthogonal(self.target_model_state[4], np.sqrt(2))
        init_orthogonal(self.target_model_state[7], np.sqrt(2))

        for param in self.target_model_state.parameters():
            param.requires_grad = False

        self.target_model_action = nn.Sequential(
            nn.Linear(action_dim, self.feature_dim // 2),
            nn.ELU(),
            nn.Linear(self.feature_dim // 2, self.feature_dim // 2),
            nn.ELU(),
        )
        init_orthogonal(self.target_model_action[0], np.sqrt(2))
        init_orthogonal(self.target_model_action[2], np.sqrt(2))

        for param in self.target_model_action.parameters():
            param.requires_grad = False

        self.target_model = nn.Linear(self.feature_dim + self.feature_dim // 2, self.feature_dim)
        init_orthogonal(self.target_model, np.sqrt(2))

        for param in self.target_model.parameters():
            param.requires_grad = False

        self.model_state = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(fc_inputs_count, self.feature_dim),
            nn.ELU()
        )
        init_orthogonal(self.model_state[0], np.sqrt(2))
        init_orthogonal(self.model_state[2], np.sqrt(2))
        init_orthogonal(self.model_state[4], np.sqrt(2))
        init_orthogonal(self.model_state[7], np.sqrt(2))

        self.model_action = nn.Sequential(
            nn.Linear(action_dim, self.feature_dim // 2),
            nn.ELU(),
            nn.Linear(self.feature_dim // 2, self.feature_dim // 2),
            nn.ELU()
        )
        init_orthogonal(self.model_action[0], np.sqrt(2))
        init_orthogonal(self.model_action[2], np.sqrt(2))

        self.model = nn.Sequential(
            nn.Linear(self.feature_dim + self.feature_dim // 2, self.feature_dim),
            nn.ELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        init_orthogonal(self.model[0], np.sqrt(2))
        init_orthogonal(self.model[2], np.sqrt(2))
        init_orthogonal(self.model[4], np.sqrt(2))

    def prepare_input(self, state, action):
        action = action - self.action_average.mean
        state = state - self.state_average.mean
        return state[:, 0, :, :].unsqueeze(1), action

    def forward(self, state, action):
        s, a = self.prepare_input(state, action)

        ts, ta = self.target_model_state(s), self.target_model_action(a)
        ps, pa = self.model_state(s), self.model_action(a)

        p = torch.cat([ps, pa], dim=1)
        t = torch.cat([ts, ta], dim=1)

        predicted_code = self.model(p)
        target_code = self.target_model(t)
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
        loss = loss.sum() / mask.sum()

        analytic = RNDAnalytic()
        analytic.update(loss_prediction=loss.unsqueeze(-1).detach())

        return loss

    def update_state_average(self, state, action):
        self.action_average.update(action)
        self.state_average.update(state)
