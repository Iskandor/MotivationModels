import torch
import torch.nn as nn

from modules import init


class ForwardModelAtari(nn.Module):
    def __init__(self, encoder, feature_dim, action_dim):
        super(ForwardModelAtari, self).__init__()

        self.encoder = encoder
        self.feature_dim = feature_dim
        self.action_dim = action_dim

        self.layers_forward_model = [
            nn.Linear(self.feature_dim + self.action_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        ]

        init(self.layers_forward_model[0], 0.1)
        init(self.layers_forward_model[2], 0.01)
        init(self.layers_forward_model[4], 0.01)

        self.forward_model = nn.Sequential(*self.layers_forward_model)

    def forward(self, state, action):
        x = self.encoder(state)
        x = torch.cat([x, action], dim=1)

        predicted_state = self.forward_model(x)
        return predicted_state

    def error(self, state, action, next_state):
        with torch.no_grad():
            prediction = self(state, action)
            target = self.encoder(next_state)
            error = torch.mean(torch.pow(prediction.view(prediction.shape[0], -1) - target.view(next_state.shape[0], -1), 2), dim=1).unsqueeze(1)

        return error

    def loss_function(self, state, action, next_state):
        loss = nn.functional.mse_loss(self(state, action), self.encoder(next_state).detach())
        return loss
