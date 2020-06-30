import abc

import torch


class MetaLearnerModel(torch.nn.Module):
    @abc.abstractmethod
    def __init__(self, state_dim, action_dim):
        super(MetaLearnerModel, self).__init__()

    @abc.abstractmethod
    def forward(self, state, action):
        raise NotImplementedError


class MetaLearnerMotivation:
    def __init__(self, network, forward_model, state_dim, action_dim, lr, weight_decay=0, variant='A'):
        self._forward_model = forward_model
        self._network = network(state_dim, action_dim)
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr, weight_decay=weight_decay)
        self._variant = variant

    def train(self, state0, action, state1):
        self._optimizer.zero_grad()
        error = self._forward_model.error(state0, action, state1)
        loss = torch.nn.functional.mse_loss(self._network(state0, action), error)
        loss.backward()
        self._optimizer.step()

        self._forward_model.train(state0, action, state1)

    def error(self, state0, action):
        with torch.no_grad():
            error = self._network(state0, action).detach().detach()
        return error

    def reward(self, state0, action, state1, eta=1.0):
        sigma = 1e-2
        k = 1
        error = self._forward_model.error(state0, action, state1)
        error_estimate = self.error(state0, action)

        reward = None
        if self._variant == 'A':
            mask = torch.gt(torch.abs(error - error_estimate), torch.ones_like(error) * sigma).type(torch.float32)
            reward = torch.tanh(error / error_estimate + error_estimate / error - 2) * mask

        if self._variant == 'B':
            reward = torch.exp(k * torch.abs(error - error_estimate)) - 1

        reward = torch.max(reward, self._forward_model.reward(error))

        return reward * eta

    def save(self, path):
        self._forward_model.save(path)
        torch.save(self._network.state_dict(), path + '_mc.pth')

    def load(self, path):
        self._forward_model.load(path)
        self._network.load_state_dict(torch.load(path + '_mc.pth'))

