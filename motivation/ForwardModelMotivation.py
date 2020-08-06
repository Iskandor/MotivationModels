import abc
import torch
from algorithms.ReplayBuffer import ModelReplayBuffer


class ForwardModel(torch.nn.Module):
    @abc.abstractmethod
    def __init__(self, state_dim, action_dim):
        super(ForwardModel, self).__init__()

    @abc.abstractmethod
    def forward(self, state, action):
        raise NotImplementedError


class ForwardModelMotivation:
    def __init__(self, network, state_dim, action_dim, lr, memory_size, sample_size, weight_decay=0, eta=1):
        self._network = network(state_dim, action_dim)
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr, weight_decay=weight_decay)
        self._memory = ModelReplayBuffer(memory_size)
        self._sample_size = sample_size
        self._eta = eta

    def train(self, state0, action, state1):
        self._memory.add(state0, action, state1)

        if len(self._memory) > self._sample_size:
            sample = self._memory.sample(self._sample_size)

            states = torch.stack(sample.state)
            next_states = torch.stack(sample.next_state)
            actions = torch.stack(sample.action)

            self._optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(self._network(states, actions), next_states)
            loss.backward()
            self._optimizer.step()

    def error(self, state0, action, state1):
        with torch.no_grad():
            prediction = self._network(state0, action).detach()
            dim = len(prediction.shape) - 1
            error = torch.mean(torch.pow(prediction - state1, 2), dim=dim).unsqueeze(dim)

        return error

    def reward(self, state0=None, action=None, state1=None, error=None):
        if error is None:
            reward = torch.tanh(self.error(state0, action, state1))
        else:
            reward = torch.tanh(error)

        return reward * self._eta

    def save(self, path):
        torch.save(self._network.state_dict(), path + '_fm.pth')

    def load(self, path):
        self._network.load_state_dict(torch.load(path + '_fm.pth'))
