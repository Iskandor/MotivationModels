import torch
import torch.nn as nn
import numpy as np

from analytic.RNDAnalytic import RNDAnalytic
from modules import init_orthogonal
from modules.encoders.EncoderAtari import ST_DIMEncoderAtari


class FWDModelAtari(nn.Module):
    def __init__(self, input_shape, feature_dim, action_dim, config):
        super(FWDModelAtari, self).__init__()

        self.input_shape = input_shape
        self.encoder = ST_DIMEncoderAtari(input_shape, feature_dim, config)

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

    def forward(self, state, action):
        encoded_state = self.encoder(state)
        predicted_state = self.forward_model(torch.cat((encoded_state, action), dim=1))
        return predicted_state

    def error(self, state, action, next_state):
        with torch.no_grad():
            predicted_state = self(state, action)
            target = self.encoder(next_state)
            error = torch.mean(torch.pow(predicted_state.view(predicted_state.shape[0], -1) - target.view(next_state.shape[0], -1), 2), dim=1).unsqueeze(1)

        return error

    def loss_function(self, state, action, next_state):
        loss_target, _, loss_target_norm = self.encoder.loss_function_crossentropy(state, next_state)
        loss_target_norm *= 1e-4

        predicted_state = self(state, action)
        target = self.encoder(next_state)
        fwd_loss = nn.functional.mse_loss(predicted_state, target)

        #update analytic
        RNDAnalytic().update(loss_prediction=fwd_loss)
        return loss_target + loss_target_norm + fwd_loss
