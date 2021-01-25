import torch
from torch import nn
from torch.nn import *

from algorithms.DDPG import DDPGCritic, DDPGActor
from motivation.MateLearnerMotivation import MetaLearnerModel


class Critic(DDPGCritic):
    def __init__(self, state_dim, action_dim, config):
        super(Critic, self).__init__(state_dim, action_dim)

        self._hidden0 = nn.Linear(state_dim, config.critic_h1)
        self._hidden1 = nn.Linear(config.critic_h1 + action_dim, config.critic_h2)
        self._output = nn.Linear(config.critic_h2, 1)

        self.init()

    def forward(self, state, action):
        x = state
        x = torch.relu(self._hidden0(x))
        x = torch.cat([x, action], 1)
        x = torch.relu(self._hidden1(x))
        value = self._output(x)
        return value

    def init(self):
        nn.init.xavier_uniform_(self._hidden0.weight)
        nn.init.xavier_uniform_(self._hidden1.weight)
        nn.init.uniform_(self._output.weight, -3e-3, 3e-3)


class Actor(DDPGActor):
    def __init__(self, state_dim, action_dim, config):
        super(Actor, self).__init__(state_dim, action_dim)

        self._hidden0 = nn.Linear(state_dim, config.actor_h1)
        self._hidden1 = nn.Linear(config.actor_h1, config.actor_h2)
        self._output = nn.Linear(config.actor_h2, action_dim)

        self.init()

    def forward(self, state):
        x = state
        x = torch.relu(self._hidden0(x))
        x = torch.relu(self._hidden1(x))
        policy = torch.tanh(self._output(x))
        return policy

    def init(self):
        nn.init.xavier_uniform_(self._hidden0.weight)
        nn.init.xavier_uniform_(self._hidden1.weight)
        nn.init.uniform_(self._output.weight, -3e-1, 3e-1)


class MetaLearnerNetwork(MetaLearnerModel):
    def __init__(self, state_dim, action_dim, config):
        super(MetaLearnerNetwork, self).__init__(state_dim, action_dim, config)

        self.layers = [
            Linear(in_features=state_dim + action_dim, out_features=config.metacritic_h1, bias=True),
            LeakyReLU(),
            Linear(in_features=config.metacritic_h1, out_features=config.metacritic_h1, bias=True),
            LeakyReLU(),
            Linear(in_features=config.metacritic_h1, out_features=config.metacritic_h2, bias=True),
            LeakyReLU(),
            Linear(in_features=config.metacritic_h2, out_features=config.metacritic_h2, bias=True),
            LeakyReLU(),
            Linear(in_features=config.metacritic_h2, out_features=1, bias=True),
            LeakyReLU()
        ]

        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.xavier_uniform_(self.layers[2].weight)
        nn.init.xavier_uniform_(self.layers[4].weight)
        nn.init.xavier_uniform_(self.layers[6].weight)
        nn.init.uniform_(self.layers[8].weight, -0.3, 0.3)

        self._model = Sequential(*self.layers)

    def forward(self, state, action):
        x = torch.cat([state, action], state.ndim - 1)
        error_estimate = self._model(x)
        return error_estimate


class SmallMetaLearnerNetwork(MetaLearnerModel):
    def __init__(self, state_dim, action_dim, config):
        super(SmallMetaLearnerNetwork, self).__init__(state_dim, action_dim, config)

        self.layers = [
            Linear(in_features=state_dim + action_dim, out_features=config.metacritic_h1, bias=True),
            Tanh(),
            Linear(in_features=config.metacritic_h1, out_features=config.metacritic_h2, bias=True),
            Tanh(),
            Linear(in_features=config.metacritic_h2, out_features=1, bias=True)
        ]

        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.xavier_uniform_(self.layers[2].weight)
        nn.init.uniform_(self.layers[4].weight, -0.3, 0.3)

        self._model = Sequential(*self.layers)

    def forward(self, state, action):
        x = torch.cat([state, action], state.ndim - 1)
        error_estimate = self._model(x)
        return error_estimate
