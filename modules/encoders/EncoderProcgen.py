import random
import time
from math import sqrt

import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms

from modules import init_orthogonal


class EncoderProcgen(nn.Module):
    def __init__(self, input_shape, feature_dim):
        super(EncoderProcgen, self).__init__()

        input_channels = input_shape[0]
        input_height = input_shape[1]
        input_width = input_shape[2]

        fc_inputs_count = 64 * (input_width // 8) * (input_height // 8)

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_inputs_count, feature_dim),
            nn.ReLU()
        )

        init_orthogonal(self.features[0], np.sqrt(2))
        init_orthogonal(self.features[2], np.sqrt(2))
        init_orthogonal(self.features[4], np.sqrt(2))
        init_orthogonal(self.features[6], np.sqrt(2))
        init_orthogonal(self.features[9], np.sqrt(2))

    def forward(self, state):
        features = self.features(state)
        return features

    def loss_function(self, state, next_state):
        loss = self.variation_prior(state) + self.stability_prior(state, next_state)
        return loss

    def variation_prior(self, state):
        sa = state[torch.randperm(state.shape[0])]
        sb = state[torch.randperm(state.shape[0])]
        variation_loss = torch.exp((self.features(sa) - self.features(sb).detach()).abs() * -1.0).mean()
        return variation_loss

    def stability_prior(self, state, next_state):
        stability_loss = (self.features(next_state) - self.features(state).detach()).abs().pow(2).mean()
        return stability_loss


class AutoEncoderProcgen(nn.Module):
    def __init__(self, input_shape, feature_dim, norm):
        super(AutoEncoderProcgen, self).__init__()

        self.norm = norm
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

        fc_inputs_count = 64 * (self.input_width // 8) * (self.input_height // 8)

        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_inputs_count, feature_dim),
            nn.ReLU()
        )

        init_orthogonal(self.encoder[0], np.sqrt(2))
        init_orthogonal(self.encoder[2], np.sqrt(2))
        init_orthogonal(self.encoder[4], np.sqrt(2))
        init_orthogonal(self.encoder[6], np.sqrt(2))
        init_orthogonal(self.encoder[9], np.sqrt(2))

        self.decoder_lin = nn.Sequential(
            nn.Linear(feature_dim, fc_inputs_count),
            nn.ReLU()
        )

        init_orthogonal(self.decoder_lin[0], np.sqrt(2))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

        init_orthogonal(self.decoder_conv[0], np.sqrt(2))
        init_orthogonal(self.decoder_conv[2], np.sqrt(2))
        init_orthogonal(self.decoder_conv[4], np.sqrt(2))
        init_orthogonal(self.decoder_conv[6], np.sqrt(2))

    def forward(self, state):
        features = self.encoder(state)
        return features

    def l1_norm(self, prediction, target):
        return torch.abs(target - prediction).sum(dim=(1, 2, 3))

    def l2_norm(self, prediction, target):
        return torch.pow(target - prediction, 2).sum(dim=(1, 2, 3)).sqrt()

    def l01_norm(self, prediction, target):
        return self.lk_norm(prediction, target, 0.1)

    def l05_norm(self, prediction, target):
        return self.lk_norm(prediction, target, 0.5)

    def lk_norm(self, prediction, target, k):
        return torch.abs(target - prediction).pow(k).sum(dim=(1, 2, 3)).pow(1 / k)

    def loss_function(self, states, next_states):
        features = self.encoder(states)
        prediction = self.decoder_lin(features).reshape(-1, 128, self.input_height // 8, self.input_width // 8)
        prediction = self.decoder_conv(prediction)

        if self.norm == 'l1':
            loss = self.l1_norm(prediction, states)
        if self.norm == 'l2':
            loss = self.l2_norm(prediction, states)
        if self.norm == 'l01':
            loss = self.l01_norm(prediction, states)
        if self.norm == 'l05':
            loss = self.l05_norm(prediction, states)

        variation_loss = self.variation_prior(states)
        stability_loss = self.stability_prior(states, next_states)
        loss = loss.mean()

        print('AE loss: {0:f} Variation prior {1:f} Stability prior: {2:f}'.format(loss.item(), variation_loss.item(), stability_loss.item()))

        return loss + variation_loss + stability_loss

    def variation_prior(self, state):
        sa = state[torch.randperm(state.shape[0])]
        sb = state[torch.randperm(state.shape[0])]
        variation_loss = torch.exp((self.encoder(sa) - self.encoder(sb).detach()).abs() * -1.0).mean()
        return variation_loss

    def stability_prior(self, state, next_state):
        stability_loss = (self.encoder(next_state) - self.encoder(state).detach()).abs().pow(2).mean()
        return stability_loss


class VAEProcgen(nn.Module):
    def __init__(self, input_shape, feature_dim, norm):
        super(VAEProcgen, self).__init__()

        self.norm = norm
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

        fc_inputs_count = 64 * (self.input_width // 8) * (self.input_height // 8)

        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        init_orthogonal(self.encoder[0], np.sqrt(2))
        init_orthogonal(self.encoder[2], np.sqrt(2))
        init_orthogonal(self.encoder[4], np.sqrt(2))
        init_orthogonal(self.encoder[6], np.sqrt(2))

        self.z_mean = nn.Sequential(
            nn.Linear(fc_inputs_count, feature_dim),
        )
        init_orthogonal(self.z_mean[0], np.sqrt(2))

        self.z_var = nn.Sequential(
            nn.Linear(fc_inputs_count, feature_dim),
        )
        init_orthogonal(self.z_var[0], np.sqrt(2))

        self.decoder_lin = nn.Sequential(
            nn.Linear(feature_dim, fc_inputs_count),
            nn.ReLU()
        )

        init_orthogonal(self.decoder_lin[0], np.sqrt(2))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

        init_orthogonal(self.decoder[0], np.sqrt(2))
        init_orthogonal(self.decoder[2], np.sqrt(2))
        init_orthogonal(self.decoder[4], np.sqrt(2))
        init_orthogonal(self.decoder[6], np.sqrt(2))

    def forward(self, state):
        x = self.encoder(state)
        mean = self.z_mean(x)
        var = self.z_var(x)

        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        features = mean + eps * std

        return features

    def l1_norm(self, prediction, target):
        return torch.abs(target - prediction).sum(dim=(1, 2, 3))

    def l2_norm(self, prediction, target):
        return torch.pow(target - prediction, 2).sum(dim=(1, 2, 3)).sqrt()

    def l01_norm(self, prediction, target):
        return self.lk_norm(prediction, target, 0.1)

    def l05_norm(self, prediction, target):
        return self.lk_norm(prediction, target, 0.5)

    def lk_norm(self, prediction, target, k):
        return torch.abs(target - prediction).pow(k).sum(dim=(1, 2, 3)).pow(1 / k)

    def loss_function(self, state, next_states):
        x = self.encoder(state)
        mean = self.z_mean(x)
        var = self.z_var(x)

        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        features = mean + eps * std

        prediction = self.decoder_lin(features)
        prediction = self.decoder(prediction.view(-1, 128, self.input_height // 8, self.input_width // 8))

        if self.norm == 'l1':
            BCE = self.l1_norm(prediction, state)
        if self.norm == 'l2':
            BCE = self.l2_norm(prediction, state)
        if self.norm == 'l01':
            BCE = self.l01_norm(prediction, state)
        if self.norm == 'l05':
            BCE = self.l05_norm(prediction, state)

        KLD = -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp(), dim=1)

        return BCE.mean() + KLD.mean()


class DDMEncoderProcgen(nn.Module):
    def __init__(self, input_shape, action_dim, feature_dim):
        super(DDMEncoderProcgen, self).__init__()

        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

        fc_inputs_count = 64 * (self.input_width // 8) * (self.input_height // 8)

        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_inputs_count, feature_dim)
        )

        init_orthogonal(self.encoder[0], np.sqrt(2))
        init_orthogonal(self.encoder[2], np.sqrt(2))
        init_orthogonal(self.encoder[4], np.sqrt(2))
        init_orthogonal(self.encoder[6], np.sqrt(2))
        init_orthogonal(self.encoder[9], np.sqrt(2))

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

        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim + feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, action_dim),
            nn.Tanh()
        )

        init_orthogonal(self.inverse_model[0], np.sqrt(2))
        init_orthogonal(self.inverse_model[2], np.sqrt(2))
        init_orthogonal(self.inverse_model[4], np.sqrt(2))

        self.decoder_lin = nn.Sequential(
            nn.Linear(feature_dim, fc_inputs_count),
            nn.ReLU()
        )

        init_orthogonal(self.decoder_lin[0], np.sqrt(2))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

        init_orthogonal(self.decoder_conv[0], np.sqrt(2))
        init_orthogonal(self.decoder_conv[2], np.sqrt(2))
        init_orthogonal(self.decoder_conv[4], np.sqrt(2))
        init_orthogonal(self.decoder_conv[6], np.sqrt(2))

    def forward(self, state):
        return self.encoder(state)

    def predict(self, state, action, next_state):
        features = self.encoder(state)
        next_features = self.encoder(next_state)

        x = torch.cat([features, action], dim=1)
        next_feature_pred = self.forward_model(x)

        x = torch.cat([features, next_features], dim=1)
        action_pred = self.inverse_model(x)

        state_pred = self.decoder_lin(features).reshape(-1, 64, self.input_height // 8, self.input_width // 8)
        state_pred = self.decoder_conv(state_pred)

        next_state_pred = self.decoder_lin(next_feature_pred).reshape(-1, 64, self.input_height // 8, self.input_width // 8)
        next_state_pred = self.decoder_conv(next_state_pred)

        return state_pred, next_feature_pred, next_state_pred, action_pred

    def loss_function(self, states, actions, next_states):
        state_pred, next_feature_pred, next_state_pred, action_pred = self.predict(states, actions, next_states)
        next_feature_target = self.encoder(next_states)

        ae_loss = nn.functional.mse_loss(state_pred, states)
        next_feature_loss = nn.functional.mse_loss(next_feature_pred, next_feature_target.detach())
        forward_model_loss = nn.functional.mse_loss(next_state_pred, next_states)
        inverse_model_loss = nn.functional.mse_loss(action_pred, actions)

        print('AE loss: {0:f} forward model loss {1:f} next features loss: {2:f} inverse model loss {3:f}'.format(
            ae_loss.item(), forward_model_loss.item(), next_feature_loss.item(), inverse_model_loss.item()))

        loss = ae_loss + next_feature_loss + forward_model_loss + inverse_model_loss

        return loss


class ST_DIM_CNN(nn.Module):

    def __init__(self, input_shape, feature_dim):
        super().__init__()
        self.feature_size = feature_dim
        self.hidden_size = self.feature_size

        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

        self.final_conv_size = 64 * (self.input_width // 8) * (self.input_height // 8)
        self.main = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.final_conv_size, feature_dim)
        )

        # gain = nn.init.calculate_gain('relu')
        gain = 0.5
        init_orthogonal(self.main[0], gain)
        init_orthogonal(self.main[2], gain)
        init_orthogonal(self.main[4], gain)
        init_orthogonal(self.main[6], gain)
        init_orthogonal(self.main[9], gain)

        self.local_layer_depth = self.main[4].out_channels

    def forward(self, inputs, fmaps=False):
        f5 = self.main[:6](inputs)
        out = self.main[6:](f5)

        if fmaps:
            return {
                'f5': f5.permute(0, 2, 3, 1),
                'out': out
            }
        return out


class ST_DIMEncoderProcgen(nn.Module):
    def __init__(self, input_shape, feature_dim, config):
        super(ST_DIMEncoderProcgen, self).__init__()

        self.config = config
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

        self.encoder = ST_DIM_CNN(input_shape, feature_dim)
        self.classifier1 = nn.Linear(self.encoder.hidden_size, self.encoder.local_layer_depth)  # x1 = global, x2=patch, n_channels = 32
        self.classifier2 = nn.Linear(self.encoder.local_layer_depth, self.encoder.local_layer_depth)

    def forward(self, state, fmaps=False):
        return self.encoder(state, fmaps)

    def loss_function_crossentropy(self, states, next_states):
        f_t_maps, f_t_prev_maps = self.encoder(next_states, fmaps=True), self.encoder(states, fmaps=True)

        # Loss 1: Global at time t, f5 patches at time t-1
        f_t, f_t_prev = f_t_maps['out'], f_t_prev_maps['f5']
        sy = f_t_prev.size(1)
        sx = f_t_prev.size(2)
        N = f_t.size(0)

        positive = []
        for y in range(sy):
            for x in range(sx):
                positive.append(f_t_prev[:, y, x, :].T)

        predictions = self.classifier1(f_t)
        positive = torch.stack(positive)
        logits = torch.matmul(predictions, positive)
        target = torch.arange(N).to(self.config.device).unsqueeze(0).repeat(logits.shape[0], 1)
        loss1 = nn.functional.cross_entropy(logits, target, reduction='mean')
        norm_loss1 = torch.norm(logits, p=2, dim=[1, 2]).mean()

        # Loss 2: f5 patches at time t, with f5 patches at time t-1
        f_t = f_t_maps['f5']
        predictions = []
        positive = []
        for y in range(sy):
            for x in range(sx):
                predictions.append(self.classifier2(f_t[:, y, x, :]))
                positive.append(f_t_prev[:, y, x, :].T)

        predictions = torch.stack(predictions)
        positive = torch.stack(positive)
        logits = torch.matmul(predictions, positive)
        target = torch.arange(N).to(self.config.device).unsqueeze(0).repeat(logits.shape[0], 1)
        loss2 = nn.functional.cross_entropy(logits, target, reduction='mean')
        norm_loss2 = torch.norm(logits, p=2, dim=[1, 2]).mean()

        loss = loss1 + loss2
        norm_loss = norm_loss1 + norm_loss2

        return loss, norm_loss

    def loss_function_cdist(self, states, next_states):
        f_t_maps, f_t_prev_maps = self.encoder(next_states, fmaps=True), self.encoder(states, fmaps=True)

        # Loss 1: Global at time t, f5 patches at time t-1
        f_t, f_t_prev = f_t_maps['out'], f_t_prev_maps['f5']
        sy = f_t_prev.size(1)
        sx = f_t_prev.size(2)

        N = f_t.size(0)
        target = torch.ones((N, N), device=self.config.device) - torch.eye(N, N, device=self.config.device)
        loss1 = 0.
        for y in range(sy):
            for x in range(sx):
                predictions = self.classifier1(f_t) + 1e-8
                positive = f_t_prev[:, y, x, :] + 1e-8
                logits = torch.cdist(predictions, positive, p=0.5)
                step_loss = nn.functional.mse_loss(logits, target)
                loss1 += step_loss

        loss1 = loss1 / (sx * sy)

        # Loss 2: f5 patches at time t, with f5 patches at time t-1
        f_t = f_t_maps['f5']
        loss2 = 0.
        for y in range(sy):
            for x in range(sx):
                predictions = self.classifier2(f_t[:, y, x, :]) + 1e-8
                positive = f_t_prev[:, y, x, :] + 1e-8
                logits = torch.cdist(predictions, positive, p=0.5)
                step_loss = nn.functional.mse_loss(logits, target)
                loss2 += step_loss

        loss2 = loss2 / (sx * sy)

        loss = loss1 + loss2

        return loss


class CNDVEncoderProcgen(nn.Module):
    def __init__(self, input_shape, feature_dim, config):
        super(CNDVEncoderProcgen, self).__init__()

        self.config = config
        fc_size = (input_shape[1] // 8) * (input_shape[2] // 8)

        self.layers = [
            nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),

            nn.Flatten(),

            nn.Linear(64 * fc_size, feature_dim)
        ]

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.orthogonal_(self.layers[i].weight, 2.0 ** 0.5)
                torch.nn.init.zeros_(self.layers[i].bias)

        self.encoder = nn.Sequential(*self.layers)

    def forward(self, state):
        return self.encoder(state)

    def loss_function(self, states_a, states_b, target):
        xa = states_a.clone()
        xb = states_b.clone()

        # normalise states
        # if normalise is not None:
        #     xa = normalise(xa)
        #     xb = normalise(xb)

        # states augmentation
        xa = self.augment(xa)
        xb = self.augment(xb)

        # obtain features from model
        za = self(xa)
        zb = self(xb)

        # predict close distance for similar, far distance for different states
        predicted = ((za - zb) ** 2).mean(dim=1)

        # similarity MSE loss
        loss_sim = ((target - predicted) ** 2).mean()

        # L2 magnitude regularisation
        magnitude = (za ** 2).mean() + (zb ** 2).mean()

        # care only when magnitude above 200
        loss_magnitude = torch.relu(magnitude - 200.0)

        loss = loss_sim + loss_magnitude

        return loss

    def augment(self, x):
        x = self.aug_random_apply(x, 0.5, self.aug_mask_tiles)
        x = self.aug_random_apply(x, 0.5, self.aug_noise)

        return x.detach()

    @staticmethod
    def aug_random_apply(x, p, aug_func):
        mask = (torch.rand(x.shape[0]) < p)
        mask = mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        mask = mask.float().to(x.device)
        y = (1.0 - mask) * x + mask * aug_func(x)

        return y

    @staticmethod
    def aug_mask_tiles(x, p=0.1):

        if x.shape[2] == 96:
            tile_sizes = [1, 2, 4, 8, 12, 16]
        else:
            tile_sizes = [1, 2, 4, 8, 16]

        tile_size = tile_sizes[np.random.randint(len(tile_sizes))]

        size_h = x.shape[2] // tile_size
        size_w = x.shape[3] // tile_size

        mask = (torch.rand((x.shape[0], 1, size_h, size_w)) < (1.0 - p))

        mask = torch.kron(mask, torch.ones(tile_size, tile_size))

        return x * mask.float().to(x.device)

    # uniform aditional noise
    @staticmethod
    def aug_noise(x, k=0.2):
        pointwise_noise = k * (2.0 * torch.rand(x.shape, device=x.device) - 1.0)
        return x + pointwise_noise


class BarlowTwinsEncoderProcgen(nn.Module):
    def __init__(self, input_shape, feature_dim, config):
        super(BarlowTwinsEncoderProcgen, self).__init__()

        self.config = config
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.feature_dim = feature_dim

        self.encoder = ST_DIM_CNN(input_shape, feature_dim)
        self.lam = 5e-3

        self.lam_mask = torch.maximum(torch.ones(self.feature_dim, self.feature_dim, device=self.config.device) * self.lam, torch.eye(self.feature_dim, self.feature_dim, device=self.config.device))

    def forward(self, state):
        return self.encoder(state)

    def loss_function(self, states, next_states):
        n = states.shape[0]
        d = self.feature_dim
        y_a = self.augment(states)
        y_b = self.augment(states)
        z_a = self.encoder(y_a)
        z_b = self.encoder(y_b)

        # z_a = (z_a - z_a.mean(dim=0)) / z_a.std(dim=0)
        # z_b = (z_b - z_b.mean(dim=0)) / z_b.std(dim=0)

        c = torch.matmul(z_a.t(), z_b) / n
        c_diff = (c - torch.eye(d, d, device=self.config.device)).pow(2) * self.lam_mask
        loss = c_diff.sum()

        return loss

    def augment(self, x):
        # ref = transforms.ToPILImage()(x[0])
        # ref.show()
        # transforms_train = torchvision.transforms.Compose([
        #     transforms.RandomResizedCrop(96, scale=(0.66, 1.0))])
        # transforms_train = transforms.RandomErasing(p=1)
        # print(x.max())
        ax = x + torch.randn_like(x) * 0.1
        ax = nn.functional.upsample(nn.functional.avg_pool2d(ax, kernel_size=2), scale_factor=2, mode='bilinear')
        # print(ax.max())

        # aug = transforms.ToPILImage()(ax[0])
        # aug.show()

        return ax


class VICRegEncoderProcgen(nn.Module):
    def __init__(self, input_shape, feature_dim, config):
        super(VICRegEncoderProcgen, self).__init__()

        self.config = config
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.feature_dim = feature_dim

        self.encoder = ST_DIM_CNN(input_shape, feature_dim)

    def forward(self, state):
        return self.encoder(state)

    def loss_function(self, states, next_states):
        n = states.shape[0]
        d = self.feature_dim
        # y_a = self.augment(states)
        # y_b = self.augment(states)
        z_a = self.encoder(states)
        z_b = self.encoder(next_states)

        inv_loss = nn.functional.mse_loss(z_a, z_b)

        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
        std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
        var_loss = torch.mean(nn.functional.relu(1 - std_z_a)) + torch.mean(nn.functional.relu(1 - std_z_b))

        z_a = (z_a - z_a.mean(dim=0))
        z_b = (z_b - z_b.mean(dim=0))

        cov_z_a = torch.matmul(z_a.t(), z_a) / (n - 1)
        cov_z_b = torch.matmul(z_b.t(), z_b) / (n - 1)

        cov_loss = cov_z_a.masked_select(~torch.eye(self.feature_dim, dtype=torch.bool, device=self.config.device)).pow_(2).sum() / self.feature_dim + \
                   cov_z_b.masked_select(~torch.eye(self.feature_dim, dtype=torch.bool, device=self.config.device)).pow_(2).sum() / self.feature_dim

        la = 1.
        mu = 1.
        nu = 1. / 25

        return la * inv_loss + mu * var_loss + nu * cov_loss

    def augment(self, x):
        # ref = transforms.ToPILImage()(x[0])
        # ref.show()
        # transforms_train = torchvision.transforms.Compose([
        #     transforms.RandomResizedCrop(96, scale=(0.66, 1.0))])
        # transforms_train = transforms.RandomErasing(p=1)
        # print(x.max())
        ax = x + torch.randn_like(x) * 0.1
        ax = nn.functional.upsample(nn.functional.avg_pool2d(ax, kernel_size=2), scale_factor=2, mode='bilinear')
        # print(ax.max())

        # aug = transforms.ToPILImage()(ax[0])
        # aug.show()

        return ax
