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

    def error(self, state0, action, state1):
        error = None
        with torch.no_grad():
            prediction = self._network(state0, action)
        if state0.ndim == 1:
            error = torch.nn.functional.mse_loss(prediction, state1).detach().reshape([1])
        if state0.ndim == 2:
            error = torch.zeros((state0.shape[0], 1))
            for i in range(state0.shape[0]):
                error[i] = torch.nn.functional.mse_loss(prediction[i], state1[i]).detach().reshape([1])

        return error

    def reward(self, state0, action, state1, eta=1.0):
        reward = None
        error = self.error(state0, action, state1)
        if state0.ndim == 1:
            reward = torch.tanh(error).item()
        if state0.ndim == 2:
            reward = torch.tanh(error)

        return reward * eta

    def save(self, path):
        torch.save(self._network.state_dict(), path + '_fm.pth')

    def load(self, path):
        self._network.load_state_dict(torch.load(path + '_fm.pth'))
