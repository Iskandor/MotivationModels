import torch
import torch.nn as nn
import numpy as np

from analytic.ResultCollector import ResultCollector
from modules import init_orthogonal
from modules.encoders.EncoderAtari import ST_DIMEncoderAtari


class ICMModelAtari(nn.Module):
    def __init__(self, input_shape, feature_dim, action_dim, config):
        super(ICMModelAtari, self).__init__()

        # calar that weighs the inverse model loss against the forward model loss
        self.scaling_factor = 0.2

        # encoder
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

        self.final_conv_size = 128 * (self.input_width // 8) * (self.input_height // 8)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.final_conv_size, feature_dim)
        )

        init_orthogonal(self.encoder[0], nn.init.calculate_gain('relu'))
        init_orthogonal(self.encoder[2], nn.init.calculate_gain('relu'))
        init_orthogonal(self.encoder[4], nn.init.calculate_gain('relu'))
        init_orthogonal(self.encoder[6], nn.init.calculate_gain('relu'))
        init_orthogonal(self.encoder[9], nn.init.calculate_gain('relu'))

        # dopredny model
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        init_orthogonal(self.forward_model[0], np.sqrt(2))
        init_orthogonal(self.forward_model[2], np.sqrt(2))
        init_orthogonal(self.forward_model[4], np.sqrt(2))

        # inverzny model
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim + feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, action_dim)
        )

        init_orthogonal(self.inverse_model[0], np.sqrt(2))
        init_orthogonal(self.inverse_model[2], np.sqrt(2))
        init_orthogonal(self.inverse_model[4], np.sqrt(2))

    #
    # def forward(self, state, action, next_state):
    #     encoded_state = self.encoder(state)
    #     encoded_next_state = self.encoder(next_state)
    #     predicted_action = self.inverse_model(torch.cat((encoded_state, encoded_next_state), dim=1))
    #     predicted_next_state = self.forward_model(torch.cat((encoded_state, action), dim=1))
    #     return predicted_next_state, predicted_action

    def error(self, state, action, next_state):
        with torch.no_grad():
            encoded_state = self.encoder(state)
            encoded_next_state = self.encoder(next_state)
            predicted_next_state = self.forward_model(torch.cat((encoded_state, action), dim=1))
            error = torch.mean(torch.pow(predicted_next_state.view(predicted_next_state.shape[0], -1) - encoded_next_state.view(encoded_next_state.shape[0], -1), 2), dim=1).unsqueeze(1)
        return error

    def loss_function(self, state, action, next_state):
        encoded_state = self.encoder(state)
        encoded_next_state = self.encoder(next_state)
        predicted_action = self.inverse_model(torch.cat((encoded_state, encoded_next_state), dim=1))
        # loss na predikovanu akciu
        inverse_loss = nn.functional.mse_loss(predicted_action, action.detach())

        predicted_next_state = self.forward_model(torch.cat((encoded_state, action), dim=1))
        # loss na predikovany dalsi stav
        forward_loss = nn.functional.mse_loss(predicted_next_state, encoded_next_state.detach())

        loss = (1 - self.scaling_factor) * inverse_loss + self.scaling_factor * forward_loss

        ResultCollector().update(loss_prediction=loss.unsqueeze(-1).detach().cpu(),
                                 loss_target=inverse_loss.unsqueeze(-1).detach().cpu(),
                                 loss_target_norm=torch.tensor(0, dtype=torch.float32),
                                 loss_fwd=forward_loss.unsqueeze(-1).detach().cpu())

        return loss
