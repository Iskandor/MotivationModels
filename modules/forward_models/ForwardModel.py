import random

import torch
from torch import nn
from torch.nn import *

from modules import ARCH


class ForwardModel(nn.Module):
    def __init__(self, input_shape, action_dim, config, arch=ARCH.robotic):
        super(ForwardModel, self).__init__()

        self.arch = arch

        self.input_shape = input_shape
        self.action_dim = action_dim

        if self.arch == ARCH.atari or self.arch == ARCH.robotic:
            self.layers_encoder, self.layers_model = self._create_model(input_shape, action_dim, arch, config)
            self.encoder = Sequential(*self.layers_encoder)
            self.model = Sequential(*self.layers_model)
        else:
            self.layers = self._create_model(input_shape, action_dim, arch, config)
            self.model = Sequential(*self.layers)

    def forward(self, state, action):
        predicted_state = None
        if self.arch == ARCH.robotic:
            f = self.encoder(state).detach()
            x = torch.cat([f, action], dim=1)
            x = self.model(x)
            predicted_state = x
        if self.arch == ARCH.small_robotic:
            x = torch.cat([state, action], dim=1)
            predicted_state = self.model(x)
        if self.arch == ARCH.aeris:
            if state.ndim == 3:
                a = action.unsqueeze(2).repeat(1, 1, state.shape[2])
                x = torch.cat([state, a], dim=1)
            if state.ndim == 2:
                a = action.unsqueeze(1).repeat(1, state.shape[1])
                x = torch.cat([state, a], dim=0).unsqueeze(0)

            predicted_state = self.model(x).squeeze(0)
        if self.arch == ARCH.atari:
            f = self.encoder(state)
            a = (action.float() / self.action_dim).repeat(1, f.shape[1])
            x = torch.cat([f, a], dim=1)
            x = self.model(x)
            predicted_state = x

        return predicted_state

    def encode(self, state):
        return self.encoder(state)

    def error(self, state, action, next_state):
        with torch.no_grad():
            dim = action.ndim - 1
            prediction = self(state, action)

            if self.arch == ARCH.atari or self.arch == ARCH.robotic:
                target = self.encoder(next_state)
                error = torch.mean(torch.pow(prediction - target, 2), dim=dim).unsqueeze(dim)
            else:
                error = torch.mean(torch.pow(prediction.flatten(dim) - next_state.flatten(dim), 2), dim=dim).unsqueeze(dim)

        return error

    def loss_function(self, state, action, next_state):
        if self.arch == ARCH.atari or self.arch == ARCH.robotic:
            loss = nn.functional.mse_loss(self(state, action), self.encoder(next_state).detach()) + self.variation_prior(state) + self.stability_prior(state, next_state)
        else:
            loss = nn.functional.mse_loss(self(state, action), next_state)
        return loss

    def variation_prior(self, state):
        sa = state[torch.randperm(state.shape[0])]
        sb = state[torch.randperm(state.shape[0])]
        variation_loss = torch.exp((self.encoder(sa) - self.encoder(sb)).abs() * -1.0).mean()
        return variation_loss

    def stability_prior(self, state, next_state):
        stability_loss = (self.encoder(next_state) - self.encoder(state)).abs().pow(2).mean()
        return stability_loss

    def distance_conservation(self, state):
        sa = state[torch.randperm(state.shape[0])]
        sb = state[torch.randperm(state.shape[0])]
        esa = self.encoder(sa)
        esb = self.encoder(sb)
        d1 = (sa - sb).pow(2).sum().sqrt()
        d2 = (esa - esb).pow(2).sum().sqrt()

        distance_conservation_loss = nn.functional.mse_loss(d2, d1)
        return distance_conservation_loss

    def _create_model(self, input_shape, action_dim, arch, config):
        layers = None
        if arch == ARCH.small_robotic:
            layers = self._small_robotic_arch(input_shape, action_dim, config)
        if arch == ARCH.robotic:
            layers = self._robotic_arch(input_shape, action_dim, config)
        if arch == ARCH.aeris:
            layers = self._aeris_arch(input_shape, action_dim, config)
        if arch == ARCH.atari:
            layers = self._atari_arch(input_shape, action_dim, config)

        return layers

    def _small_robotic_arch(self, input_shape, action_dim, config):
        layers = [
            Linear(in_features=input_shape + action_dim, out_features=config.forward_model_h1, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h1, out_features=config.forward_model_h2, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h2, out_features=input_shape, bias=True)
        ]

        nn.init.xavier_uniform_(layers[0].weight)
        nn.init.xavier_uniform_(layers[2].weight)
        nn.init.uniform_(layers[4].weight, -0.3, 0.3)

        return layers

    def _robotic_arch(self, input_shape, action_dim, config):
        layers_encoder = [
            Linear(in_features=input_shape, out_features=input_shape * 10, bias=True),
            LeakyReLU(),
            Linear(in_features=input_shape * 10, out_features=input_shape * 10, bias=True),
            LeakyReLU(),
            Linear(in_features=input_shape * 10, out_features=input_shape * 5, bias=True),
            LeakyReLU(),
            Linear(in_features=input_shape * 5, out_features=input_shape * 5, bias=True),
            LeakyReLU(),
            Linear(in_features=input_shape * 5, out_features=action_dim, bias=True)
        ]

        nn.init.xavier_uniform_(layers_encoder[0].weight)
        nn.init.xavier_uniform_(layers_encoder[2].weight)
        nn.init.xavier_uniform_(layers_encoder[4].weight)
        nn.init.xavier_uniform_(layers_encoder[6].weight)
        nn.init.xavier_uniform_(layers_encoder[8].weight)

        layers_model = [
            Linear(in_features=action_dim + action_dim, out_features=config.forward_model_h1, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h1, out_features=config.forward_model_h2, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h2, out_features=action_dim, bias=True)
        ]

        nn.init.xavier_uniform_(layers_model[0].weight)
        nn.init.xavier_uniform_(layers_model[2].weight)
        nn.init.uniform_(layers_model[4].weight, -0.3, 0.3)

        return layers_encoder, layers_model

    def _aeris_arch(self, input_shape, action_dim, config):
        channels = input_shape[0]

        layers = [
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

        nn.init.xavier_uniform_(layers[0].weight)
        nn.init.xavier_uniform_(layers[2].weight)
        nn.init.xavier_uniform_(layers[4].weight)
        nn.init.xavier_uniform_(layers[6].weight)
        nn.init.xavier_uniform_(layers[8].weight)
        nn.init.uniform_(layers[10].weight, -0.3, 0.3)

        return layers

    def _atari_arch(self, input_shape, action_dim, config):
        channels = input_shape[0]

        layers_encoder = [
            nn.Conv2d(channels, 32, kernel_size=7, stride=3, padding=2),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, kernel_size=5, stride=3, padding=0),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0),

            nn.Flatten()
        ]

        nn.init.xavier_uniform_(layers_encoder[0].weight)
        nn.init.xavier_uniform_(layers_encoder[2].weight)
        nn.init.xavier_uniform_(layers_encoder[4].weight)
        nn.init.xavier_uniform_(layers_encoder[6].weight)
        nn.init.xavier_uniform_(layers_encoder[8].weight)
        nn.init.xavier_uniform_(layers_encoder[10].weight)

        layers_model = [
            Linear(in_features=256, out_features=config.forward_model_h1, bias=True),
            LeakyReLU(),
            Linear(in_features=config.forward_model_h1, out_features=config.forward_model_h1, bias=True),
            LeakyReLU(),
            Linear(in_features=config.forward_model_h1, out_features=config.forward_model_h2, bias=True),
            LeakyReLU(),
            Linear(in_features=config.forward_model_h2, out_features=config.forward_model_h2, bias=True),
            LeakyReLU(),
            Linear(in_features=config.forward_model_h2, out_features=128, bias=True)
        ]

        nn.init.xavier_uniform_(layers_model[0].weight)
        nn.init.xavier_uniform_(layers_model[2].weight)
        nn.init.xavier_uniform_(layers_model[4].weight)
        nn.init.xavier_uniform_(layers_model[6].weight)
        nn.init.xavier_uniform_(layers_model[8].weight)

        return layers_encoder, layers_model

#AE arch
# def _atari_arch(self, input_shape, action_dim, config):
#     channels = input_shape[0]
#
#     layers_encoder = [
#         nn.Conv2d(channels, 64, kernel_size=7, stride=3, padding=2),
#         nn.LeakyReLU(),
#
#         nn.Conv2d(64, 64, kernel_size=5, stride=3, padding=0),
#         nn.LeakyReLU(),
#
#         nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0),
#         nn.LeakyReLU(),
#
#         nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
#         nn.LeakyReLU(),
#
#         nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
#         nn.LeakyReLU(),
#
#         nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0),
#         nn.LeakyReLU(),
#
#         nn.Flatten()
#     ]
#
#     nn.init.xavier_uniform_(layers_encoder[0].weight)
#     nn.init.xavier_uniform_(layers_encoder[2].weight)
#     nn.init.xavier_uniform_(layers_encoder[4].weight)
#     nn.init.xavier_uniform_(layers_encoder[6].weight)
#     nn.init.xavier_uniform_(layers_encoder[8].weight)
#     nn.init.xavier_uniform_(layers_encoder[10].weight)
#
#     layers_model = [
#         nn.Linear(512, 256),
#         nn.LeakyReLU(),
#         nn.Linear(256, 256),
#         nn.LeakyReLU(),
#     ]
#
#     nn.init.xavier_uniform_(layers_model[0].weight)
#     nn.init.xavier_uniform_(layers_model[2].weight)
#
#     layers_decoder = [
#         nn.ConvTranspose2d(256, 256, kernel_size=2, stride=1, padding=0),
#         nn.LeakyReLU(),
#
#         nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=0),
#         nn.LeakyReLU(),
#
#         nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=0),
#         nn.LeakyReLU(),
#
#         nn.ConvTranspose2d(128, 64, kernel_size=5, stride=1, padding=0),
#         nn.LeakyReLU(),
#
#         nn.ConvTranspose2d(64, 64, kernel_size=5, stride=3, padding=0),
#         nn.LeakyReLU(),
#
#         nn.ConvTranspose2d(64, channels, kernel_size=7, stride=3, padding=2),
#     ]
#
#     nn.init.xavier_uniform_(layers_decoder[0].weight)
#     nn.init.xavier_uniform_(layers_decoder[2].weight)
#     nn.init.xavier_uniform_(layers_decoder[4].weight)
#     nn.init.xavier_uniform_(layers_decoder[6].weight)
#     nn.init.xavier_uniform_(layers_decoder[8].weight)
#     nn.init.xavier_uniform_(layers_decoder[10].weight)
#
#     return layers_encoder, layers_model, layers_decoder