import copy
import math

import torch
from torch import nn
from torch.nn import *


class VAE(nn.Module):
    def __init__(self, state_space_dim, latent_space_dim):
        super().__init__()

        self._state_space_dim = state_space_dim
        self._latent_space_dim = latent_space_dim

        depth = math.floor(math.log2(state_space_dim) - math.log2(latent_space_dim))

        encoder_layers = []
        decoder_layers = []
        out_features = state_space_dim

        for i in range(depth):
            in_features = out_features
            out_features = out_features // 2
            encoder_layers.append(Linear(in_features, out_features, bias=True))
            encoder_layers.append(ReLU())
            decoder_layers.insert(0, Linear(out_features, in_features, bias=True))
            decoder_layers.insert(1, ReLU())

        decoder_layers = decoder_layers[:-1]

        self._encoder = Sequential(*encoder_layers)
        self._z_mean = Linear(out_features, latent_space_dim)
        self._z_var = Linear(out_features, latent_space_dim)
        self._decoder = Sequential(*decoder_layers)

    def encode(self, state):
        x = self._encoder(state)
        return self._z_mean(x), self._z_var(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self._decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self._state_space_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, x):
        mu, logvar = self.encode(x)
        recon_x = self.decode(self.reparameterize(mu, logvar))

        BCE = functional.mse_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD