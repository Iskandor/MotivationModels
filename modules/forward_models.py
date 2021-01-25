import math

import torch
from torch import nn
from torch.nn import *


class ForwardModel(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(ForwardModel, self).__init__()

        self.layers = [
            Linear(in_features=state_dim + action_dim, out_features=config.forward_model_h1, bias=True),
            LeakyReLU(),
            Linear(in_features=config.forward_model_h1, out_features=config.forward_model_h1, bias=True),
            LeakyReLU(),
            Linear(in_features=config.forward_model_h1, out_features=config.forward_model_h2, bias=True),
            LeakyReLU(),
            Linear(in_features=config.forward_model_h2, out_features=config.forward_model_h2, bias=True),
            LeakyReLU(),
            Linear(in_features=config.forward_model_h2, out_features=state_dim, bias=True)
        ]

        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.xavier_uniform_(self.layers[2].weight)
        nn.init.xavier_uniform_(self.layers[4].weight)
        nn.init.xavier_uniform_(self.layers[6].weight)
        nn.init.uniform_(self.layers[8].weight, -0.3, 0.3)

        self._model = Sequential(*self.layers)

    def forward(self, state, action):
        x = torch.cat([state, action], state.ndim - 1)
        predicted_state = self._model(x)
        return predicted_state

    def loss_function(self, state, action, next_state):
        return nn.functional.mse_loss(self(state, action), next_state)


class SmallForwardModel(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(SmallForwardModel, self).__init__()

        self.layers = [
            Linear(in_features=state_dim + action_dim, out_features=config.forward_model_h1, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h1, out_features=config.forward_model_h2, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h2, out_features=state_dim, bias=True)
        ]

        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.xavier_uniform_(self.layers[2].weight)
        nn.init.uniform_(self.layers[4].weight, -0.3, 0.3)

        self._model = Sequential(*self.layers)

    def forward(self, state, action):
        x = torch.cat([state, action], state.ndim - 1)
        predicted_state = self._model(x)
        return predicted_state

    def loss_function(self, state, action, next_state):
        return nn.functional.mse_loss(self(state, action), next_state)


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