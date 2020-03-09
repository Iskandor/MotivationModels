import abc

import torch


class ForwardModel(torch.nn.Module):
    @abc.abstractmethod
    def __init__(self, state_dim, action_dim):
        super(ForwardModel, self).__init__()

    @abc.abstractmethod
    def forward(self, state, action):
        raise NotImplementedError


class ForwardModelMotivation:
    def __init__(self, network, state_dim, action_dim, lr, weight_decay=0):
        self._network = network(state_dim, action_dim)
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, state0, action, state1):
        self._optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(self._network(state0, action), state1)
        loss.backward()
        self._optimizer.step()

    def reward(self, state0, action, state1, eta=1.0):
        return torch.nn.functional.mse_loss(self._network(state0, action), state1).item() * eta
