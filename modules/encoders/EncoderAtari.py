import torch
import torch.nn as nn
import numpy as np

from modules import init_orthogonal


class EncoderAtari(nn.Module):
    def __init__(self, input_shape, feature_dim):
        super(EncoderAtari, self).__init__()

        input_channels = input_shape[0]
        input_height = input_shape[1]
        input_width = input_shape[2]

        fc_inputs_count = 128 * (input_width // 8) * (input_height // 8)

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
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


class AutoEncoderAtari(nn.Module):
    def __init__(self, input_shape, feature_dim, norm):
        super(AutoEncoderAtari, self).__init__()

        self.norm = norm
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

        fc_inputs_count = 128 * (self.input_width // 8) * (self.input_height // 8)

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


class VAEAtari(nn.Module):
    def __init__(self, input_shape, feature_dim, norm):
        super(VAEAtari, self).__init__()

        self.norm = norm
        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

        fc_inputs_count = 128 * (self.input_width // 8) * (self.input_height // 8)

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


class DDMEncoderAtari(nn.Module):
    def __init__(self, input_shape, action_dim, feature_dim):
        super(DDMEncoderAtari, self).__init__()

        self.input_channels = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

        fc_inputs_count = 128 * (self.input_width // 8) * (self.input_height // 8)

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
        return self.encoder(state)

    def predict(self, state, action, next_state):
        features = self.encoder(state)
        next_features = self.encoder(next_state)

        x = torch.cat([features, action], dim=1)
        next_feature_pred = self.forward_model(x)

        x = torch.cat([features, next_features], dim=1)
        action_pred = self.inverse_model(x)

        state_pred = self.decoder_lin(features).reshape(-1, 128, self.input_height // 8, self.input_width // 8)
        state_pred = self.decoder_conv(state_pred)

        next_state_pred = self.decoder_lin(next_feature_pred).reshape(-1, 128, self.input_height // 8, self.input_width // 8)
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
