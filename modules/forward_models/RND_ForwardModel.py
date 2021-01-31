import torch
from torch import nn
from torch.nn import *

from modules import ARCH


class RND_ForwardModel(nn.Module):
    def __init__(self, input_shape, action_dim, config, arch=ARCH.robotic):
        super(RND_ForwardModel, self).__init__()

        self.arch = arch
        self.layers_model, self.layers_target = self._create_model(input_shape, action_dim, arch, config)
        self._model = Sequential(*self.layers_model)
        self._target = Sequential(*self.layers_target)

    def encode(self, state):

        if self.arch == ARCH.robotic or self.arch == ARCH.small_robotic:
            encoded_state = self._target(state).detach()
        if self.arch == ARCH.aeris:
            if state.ndim == 3:
                encoded_state = self._target(state).detach()
            if state.ndim == 2:
                encoded_state = self._target(state.unsqueeze(0)).detach()

        return encoded_state

    def forward(self, state, action):
        if self.arch == ARCH.robotic or self.arch == ARCH.small_robotic:
            x = torch.cat([self.encode(state), action], state.ndim - 1)
            predicted_state = self._model(x)
        if self.arch == ARCH.aeris:
            if state.ndim == 3:
                a = action.unsqueeze(2).repeat(1, 1, state.shape[2])
                x = torch.cat([self.encode(state), a], dim=1)
            if state.ndim == 2:
                a = action.unsqueeze(1).repeat(1, state.shape[1])
                x = torch.cat([self.encode(state), a.unsqueeze(0)], dim=1)

            predicted_state = self._model(x).squeeze(0)

        return predicted_state

    def error(self, state, action, next_state):
        with torch.no_grad():
            dim = action.ndim - 1
            prediction = self(state, action)
            error = torch.mean(torch.pow(prediction.flatten(dim) - self.encode(next_state).flatten(dim), 2), dim=dim).unsqueeze(dim)

        return error

    def loss_function(self, state, action, next_state):
        return nn.functional.mse_loss(self(state, action), self.encode(next_state))

    def _create_model(self, input_shape, action_dim, arch, config):
        layers_model = None
        layers_target = None
        if arch == ARCH.small_robotic:
            layers_model, layers_target = self._small_robotic_arch(input_shape, action_dim, config)
        if arch == ARCH.robotic:
            layers_model, layers_target = self._robotic_arch(input_shape, action_dim, config)
        if arch == ARCH.aeris:
            layers_model, layers_target = self._aeris_arch(input_shape, action_dim, config)
        if arch == ARCH.atari:
            layers_model, layers_target = self._atari_arch(input_shape, action_dim, config)

        return layers_model, layers_target

    def _small_robotic_arch(self, input_shape, action_dim, config):
        layers_model = [
            Linear(in_features=input_shape + action_dim, out_features=config.forward_model_h1, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h1, out_features=config.forward_model_h2, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h2, out_features=input_shape, bias=True)
        ]

        nn.init.xavier_uniform_(layers_model[0].weight)
        nn.init.xavier_uniform_(layers_model[2].weight)
        nn.init.uniform_(layers_model[4].weight, -0.3, 0.3)

        layers_target = [
            Linear(in_features=input_shape, out_features=config.forward_model_h1, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h1, out_features=config.forward_model_h2, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h2, out_features=input_shape, bias=True)
        ]
        nn.init.orthogonal_(layers_target[0].weight)
        nn.init.orthogonal_(layers_target[2].weight)
        nn.init.orthogonal_(layers_target[4].weight)

        return layers_model, layers_target

    def _robotic_arch(self, input_shape, action_dim, config):
        layers_model = [
            Linear(in_features=input_shape + action_dim, out_features=config.forward_model_h1, bias=True),
            LeakyReLU(),
            Linear(in_features=config.forward_model_h1, out_features=config.forward_model_h1, bias=True),
            LeakyReLU(),
            Linear(in_features=config.forward_model_h1, out_features=config.forward_model_h2, bias=True),
            LeakyReLU(),
            Linear(in_features=config.forward_model_h2, out_features=config.forward_model_h2, bias=True),
            LeakyReLU(),
            Linear(in_features=config.forward_model_h2, out_features=input_shape, bias=True)
        ]

        nn.init.xavier_uniform_(layers_model[0].weight)
        nn.init.xavier_uniform_(layers_model[2].weight)
        nn.init.xavier_uniform_(layers_model[4].weight)
        nn.init.xavier_uniform_(layers_model[6].weight)
        nn.init.uniform_(layers_model[8].weight, -0.3, 0.3)

        layers_target = [
            Linear(in_features=input_shape, out_features=config.forward_model_h1, bias=True),
            LeakyReLU(),
            Linear(in_features=config.forward_model_h1, out_features=config.forward_model_h1, bias=True),
            LeakyReLU(),
            Linear(in_features=config.forward_model_h1, out_features=config.forward_model_h2, bias=True),
            LeakyReLU(),
            Linear(in_features=config.forward_model_h2, out_features=config.forward_model_h2, bias=True),
            LeakyReLU(),
            Linear(in_features=config.forward_model_h2, out_features=input_shape, bias=True)
        ]

        nn.init.orthogonal_(layers_target[0].weight)
        nn.init.orthogonal_(layers_target[2].weight)
        nn.init.orthogonal_(layers_target[4].weight)
        nn.init.orthogonal_(layers_target[6].weight)
        nn.init.orthogonal_(layers_target[8].weight)

        return layers_model, layers_target

    def _aeris_arch(self, input_shape, action_dim, config):
        channels = input_shape[0]

        layers_model = [
            nn.Conv1d(channels + action_dim, config.forward_model_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(config.forward_model_kernels_count, config.forward_model_kernels_count, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(config.forward_model_kernels_count, config.forward_model_kernels_count, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(config.forward_model_kernels_count, config.forward_model_kernels_count, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(config.forward_model_kernels_count, config.forward_model_kernels_count, kernel_size=8, stride=4, padding=2, output_padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(config.forward_model_kernels_count, channels, kernel_size=3, stride=1, padding=1)
        ]

        nn.init.xavier_uniform_(layers_model[0].weight)
        nn.init.xavier_uniform_(layers_model[2].weight)
        nn.init.xavier_uniform_(layers_model[4].weight)
        nn.init.xavier_uniform_(layers_model[6].weight)
        nn.init.xavier_uniform_(layers_model[8].weight)
        nn.init.uniform_(layers_model[10].weight, -0.3, 0.3)

        layers_target = [
            nn.Conv1d(channels, config.forward_model_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(config.forward_model_kernels_count, config.forward_model_kernels_count, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(config.forward_model_kernels_count, config.forward_model_kernels_count, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(config.forward_model_kernels_count, config.forward_model_kernels_count, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(config.forward_model_kernels_count, config.forward_model_kernels_count, kernel_size=8, stride=4, padding=2, output_padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(config.forward_model_kernels_count, channels, kernel_size=3, stride=1, padding=1)
        ]

        nn.init.orthogonal_(layers_target[0].weight)
        nn.init.orthogonal_(layers_target[2].weight)
        nn.init.orthogonal_(layers_target[4].weight)
        nn.init.orthogonal_(layers_target[6].weight)
        nn.init.orthogonal_(layers_target[8].weight)
        nn.init.orthogonal_(layers_target[10].weight)

        return layers_model, layers_target