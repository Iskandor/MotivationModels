import torch
from torch import nn

from modules import ARCH
from modules.forward_models.ForwardModel import ForwardModel


class MetaCriticRobotic(nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(MetaCriticRobotic, self).__init__()

        self.forward_model = ForwardModel(input_shape, action_dim, config, ARCH.robotic)

        self.layers_metacritic = [
            nn.Linear(in_features=action_dim + action_dim + action_dim, out_features=config.metacritic_h1, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=config.metacritic_h1, out_features=config.metacritic_h1, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=config.metacritic_h1, out_features=config.metacritic_h2, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=config.metacritic_h2, out_features=config.metacritic_h2, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=config.metacritic_h2, out_features=1, bias=True)
        ]

        nn.init.xavier_uniform_(self.layers_metacritic[0].weight)
        nn.init.xavier_uniform_(self.layers_metacritic[2].weight)
        nn.init.xavier_uniform_(self.layers_metacritic[4].weight)
        nn.init.xavier_uniform_(self.layers_metacritic[6].weight)
        nn.init.uniform_(self.layers_metacritic[8].weight, -0.3, 0.3)

        self.metacritic = nn.Sequential(*self.layers_metacritic)

    def forward(self, state, action):
        f = self.forward_model.encoder(state).detach()
        x = torch.cat([f, action], dim=1)
        predicted_state = self.forward_model.model(x)
        estimated_error = self.metacritic(x)
        return predicted_state, estimated_error

    def error(self, state, action, next_state):
        return self.forward_model.error(state, action, next_state)

    def error_estimate(self, state, action):
        fs0 = self.forward_model.encoder(state).detach()
        x = torch.cat([fs0, action], dim=1)
        fs1 = self.forward_model.model(x).detach()
        x = torch.cat([fs0, action, fs1], dim=1)
        estimated_error = self.metacritic(x)
        return estimated_error

    def loss_function(self, state, action, next_state):
        fm_loss = self.forward_model.loss_function(state, action, next_state)
        mc_loss = nn.functional.mse_loss(self.error_estimate(state, action), self.error(state, action, next_state), reduction='sum')
        return fm_loss + mc_loss
