import math

import torch
from torch import nn
from torch.nn import *

from modules import ARCH


class ForwardModel(nn.Module):
    def __init__(self, input_shape, action_dim, config, arch=ARCH.robotic):
        super(ForwardModel, self).__init__()

        self.arch = arch

        if self.arch == ARCH.atari:
            self.layers_encoder, self.layers_model, self.layers_decoder = self._create_model(input_shape, action_dim, arch, config)
            self._encoder = Sequential(*self.layers_encoder)
            self._model = Sequential(*self.layers_model)
            self._decoder = Sequential(*self.layers_decoder)
        else:
            self.layers = self._create_model(input_shape, action_dim, arch, config)
            self._model = Sequential(*self.layers)

    def forward(self, state, action):
        predicted_state = None
        if self.arch == ARCH.robotic or self.arch == ARCH.small_robotic:
            x = torch.cat([state, action], state.ndim - 1)
            predicted_state = self._model(x)
        if self.arch == ARCH.aeris:
            if state.ndim == 3:
                a = action.unsqueeze(2).repeat(1, 1, state.shape[2])
                x = torch.cat([state, a], dim=1)
            if state.ndim == 2:
                a = action.unsqueeze(1).repeat(1, state.shape[1])
                x = torch.cat([state, a], dim=0).unsqueeze(0)

            predicted_state = self._model(x).squeeze(0)
        if self.arch == ARCH.atari:
            x = self._encoder(state)
            x = torch.cat([x, action.float().repeat(1, x.shape[1])], dim=1)
            x = self._model(x).unsqueeze(2).unsqueeze(3)
            x = self._decoder(x)
            predicted_state = x

        return predicted_state

    def error(self, state, action, next_state):
        with torch.no_grad():
            dim = action.ndim - 1
            prediction = self(state, action)
            error = torch.mean(torch.pow(prediction.flatten(dim) - next_state.flatten(dim), 2), dim=dim).unsqueeze(dim)

        return error

    def loss_function(self, state, action, next_state):
        return nn.functional.mse_loss(self(state, action), next_state)

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
        layers = [
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

        nn.init.xavier_uniform_(layers[0].weight)
        nn.init.xavier_uniform_(layers[2].weight)
        nn.init.xavier_uniform_(layers[4].weight)
        nn.init.xavier_uniform_(layers[6].weight)
        nn.init.uniform_(layers[8].weight, -0.3, 0.3)

        return layers

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
            nn.Conv2d(channels, 64, kernel_size=7, stride=3, padding=2),
            nn.LeakyReLU(),

            nn.Conv2d(64, 64, kernel_size=5, stride=3, padding=0),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.Flatten()
        ]

        nn.init.xavier_uniform_(layers_encoder[0].weight)
        nn.init.xavier_uniform_(layers_encoder[2].weight)
        nn.init.xavier_uniform_(layers_encoder[4].weight)
        nn.init.xavier_uniform_(layers_encoder[6].weight)
        nn.init.xavier_uniform_(layers_encoder[8].weight)
        nn.init.xavier_uniform_(layers_encoder[10].weight)

        layers_model = [
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
        ]

        nn.init.xavier_uniform_(layers_model[0].weight)
        nn.init.xavier_uniform_(layers_model[2].weight)

        layers_decoder = [
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=3, padding=0),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, channels, kernel_size=7, stride=3, padding=2),
        ]

        nn.init.xavier_uniform_(layers_decoder[0].weight)
        nn.init.xavier_uniform_(layers_decoder[2].weight)
        nn.init.xavier_uniform_(layers_decoder[4].weight)
        nn.init.xavier_uniform_(layers_decoder[6].weight)
        nn.init.xavier_uniform_(layers_decoder[8].weight)
        nn.init.xavier_uniform_(layers_decoder[10].weight)

        return layers_encoder, layers_model, layers_decoder


class ResidualForwardModel(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(ResidualForwardModel, self).__init__()

        self._model = Sequential(
            Linear(in_features=state_dim + action_dim, out_features=config.forward_model_h1, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h1, out_features=config.forward_model_h1, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h1, out_features=config.forward_model_h2, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h2, out_features=config.forward_model_h2, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h2, out_features=state_dim, bias=True)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], state.ndim - 1)
        predicted_state = self._model(x) + state
        return predicted_state

    def error(self, state, action, next_state):
        with torch.no_grad():
            dim = action.ndim - 1
            prediction = self(state, action)
            error = torch.mean(torch.pow(prediction.flatten(dim) - next_state.flatten(dim), 2), dim=dim).unsqueeze(dim)

        return error

    def loss_function(self, state, action, next_state):
        return nn.functional.mse_loss(self(state, action), next_state)


class VAE_ForwardModel(nn.Module):
    def __init__(self, state_space_dim, latent_space_dim, action_space_dim):
        super().__init__()

        self._state_space_dim = state_space_dim
        self._latent_space_dim = latent_space_dim
        self._action_space_dim = action_space_dim

        # encoder_layers, decoder_layers, out_features = self._log2_arch(state_space_dim, latent_space_dim)
        self._encoder_layers, self._decoder_layers, out_features = self._step_arch(state_space_dim, latent_space_dim, 4)

        self._encoder = Sequential(*self._encoder_layers)
        self._z_mean = Linear(out_features, latent_space_dim)
        nn.init.xavier_uniform_(self._z_mean.weight)
        self._z_var = Linear(out_features, latent_space_dim)
        nn.init.xavier_uniform_(self._z_var.weight)
        self._next_z = Linear(latent_space_dim + action_space_dim, latent_space_dim)
        nn.init.xavier_uniform_(self._next_z.weight)
        self._decoder = Sequential(*self._decoder_layers)

    def _step_arch(self, state_space_dim, latent_space_dim, step):
        depth = ((state_space_dim - latent_space_dim) // step) - 1

        dims = []

        for i in range(depth):
            dims.append((state_space_dim - (i * step), state_space_dim - ((i + 1) * step)))

        encoder_layers = []
        decoder_layers = []

        for i in range(depth):
            in_features, out_features = dims[i]
            e_layer = Linear(in_features, out_features, bias=True)
            nn.init.xavier_uniform_(e_layer.weight)
            encoder_layers.append(e_layer)
            encoder_layers.append(LeakyReLU())

            out_features, in_features = dims[depth - i - 1]
            d_layer = Linear(in_features, out_features, bias=True)
            nn.init.xavier_uniform_(d_layer.weight)
            decoder_layers.append(d_layer)
            decoder_layers.append(LeakyReLU())

        _, out_features = dims[depth - 1]
        decoder_layers = decoder_layers[:-1]
        decoder_layers.insert(0, Linear(latent_space_dim, out_features, bias=True))
        decoder_layers.insert(1, LeakyReLU())

        return encoder_layers, decoder_layers, out_features

    def _log2_arch(self, state_space_dim, latent_space_dim):
        depth = math.floor(math.log2(state_space_dim) - math.log2(latent_space_dim)) - 1

        encoder_layers = []
        decoder_layers = []
        out_features = state_space_dim

        for i in range(depth):
            in_features = out_features
            out_features = out_features // 2
            e_layer = Linear(in_features, out_features, bias=True)
            nn.init.xavier_uniform_(e_layer.weight)
            encoder_layers.append(e_layer)
            encoder_layers.append(LeakyReLU())
            d_layer = Linear(out_features, in_features, bias=True)
            nn.init.xavier_uniform_(d_layer.weight)
            decoder_layers.insert(0, d_layer)
            decoder_layers.insert(1, LeakyReLU())

        decoder_layers = decoder_layers[:-1]
        decoder_layers.insert(0, Linear(latent_space_dim, out_features, bias=True))
        decoder_layers.insert(1, LeakyReLU())

        return encoder_layers, decoder_layers, out_features

    def encode(self, state):
        x = self._encoder(state)
        return self._z_mean(x), self._z_var(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self._decoder(z)
        return x

    def forward(self, state, action):
        mu, logvar = self.encode(state)
        z = self.reparameterize(mu, logvar)
        x = torch.cat([z, action], dim=state.ndim - 1)
        next_z = self._next_z(x)
        return self.decode(next_z), mu, logvar

    def loss_function(self, state, action, next_state):
        predicted_state, mu, logvar = self(state, action)
        BCE = functional.mse_loss(predicted_state, next_state)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD


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
