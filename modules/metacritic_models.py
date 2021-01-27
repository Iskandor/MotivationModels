import torch
from torch import nn


class MetaCritic(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(MetaCritic, self).__init__()

        self.layers = [
            nn.Linear(in_features=state_dim + action_dim, out_features=config.metacritic_h1, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=config.metacritic_h1, out_features=config.metacritic_h1, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=config.metacritic_h1, out_features=config.metacritic_h2, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=config.metacritic_h2, out_features=config.metacritic_h2, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=config.metacritic_h2, out_features=1, bias=True),
            nn.LeakyReLU()
        ]

        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.xavier_uniform_(self.layers[2].weight)
        nn.init.xavier_uniform_(self.layers[4].weight)
        nn.init.xavier_uniform_(self.layers[6].weight)
        nn.init.uniform_(self.layers[8].weight, -0.3, 0.3)

        self._model = nn.Sequential(*self.layers)

    def forward(self, state, action):
        x = torch.cat([state, action], state.ndim - 1)
        error_estimate = self._model(x)
        return error_estimate


class SmallMetaCritic(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(SmallMetaCritic, self).__init__()

        self.layers = [
            nn.Linear(in_features=state_dim + action_dim, out_features=config.metacritic_h1, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=config.metacritic_h1, out_features=config.metacritic_h2, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=config.metacritic_h2, out_features=1, bias=True)
        ]

        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.xavier_uniform_(self.layers[2].weight)
        nn.init.uniform_(self.layers[4].weight, -0.3, 0.3)

        self._model = nn.Sequential(*self.layers)

    def forward(self, state, action):
        x = torch.cat([state, action], state.ndim - 1)
        error_estimate = self._model(x)
        return error_estimate