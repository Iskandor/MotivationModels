import torch
import torch.nn as nn


class EncoderAeris(nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(EncoderAeris, self).__init__()
        self.channels = input_shape[0]
        self.width = input_shape[1]

        self.action_dim = action_dim

        fc_count = config.forward_model_kernels_count * self.width // 4

        self.feature_dim = config.forward_model_kernels_count

        channels = input_shape[0]

        self.layers_encoder = [
            nn.Conv1d(channels, config.forward_model_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(config.forward_model_kernels_count, config.forward_model_kernels_count, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_count, fc_count // 2),
            nn.ReLU(),
            nn.Linear(fc_count // 2, self.feature_dim),
            nn.Tanh()
        ]

        nn.init.xavier_uniform_(self.layers_encoder[0].weight)
        nn.init.xavier_uniform_(self.layers_encoder[2].weight)
        nn.init.xavier_uniform_(self.layers_encoder[5].weight)
        nn.init.xavier_uniform_(self.layers_encoder[7].weight)

        self.encoder = nn.Sequential(*self.layers_encoder)

    def forward(self, state):
        features = self.encoder(state)
        return features

    def loss_function(self, state, next_state):
        loss = self.variation_prior(state) + self.stability_prior(state, next_state)
        return loss

    def variation_prior(self, state):
        sa = state[torch.randperm(state.shape[0])]
        sb = state[torch.randperm(state.shape[0])]
        variation_loss = torch.exp((self.encoder(sa) - self.encoder(sb).detach()).abs() * -1.0).mean()
        return variation_loss

    def stability_prior(self, state, next_state):
        stability_loss = (self.encoder(next_state) - self.encoder(state).detach()).abs().pow(2).mean()
        return stability_loss
