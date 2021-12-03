import torch

from utils.RunningAverage import RunningStats


class RNDMotivation:
    def __init__(self, network, lr, eta=1, device='cpu'):
        self._network = network
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        self._eta = eta
        self._device = device
        self.reward_stats = RunningStats(1, device)

    def train(self, memory, indices):
        if indices:
            sample, size = memory.sample_batches(indices)

            for i in range(size):
                states = sample.state[i].to(self._device)

                self._optimizer.zero_grad()
                loss = self._network.loss_function(states)
                loss.backward()
                self._optimizer.step()

    def error(self, state0):
        return self._network.error(state0)

    def reward_sample(self, memory, indices):
        sample = memory.sample(indices)

        states = sample.state.to(self._device)

        return self.reward(states)

    def reward(self, state0):
        reward = self.error(state0)
        return reward * self._eta

    def update_state_average(self, state):
        self._network.update_state_average(state)

    def update_reward_average(self, reward):
        self.reward_stats.update(reward.to(self._device))


class QRNDMotivation:
    def __init__(self, network, lr, eta=1, device='cpu'):
        self._network = network
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        self._eta = eta
        self._device = device

    def train(self, memory, indices):
        if indices:
            sample = memory.sample(indices)

            states = sample.state.to(self._device)
            actions = sample.action.to(self._device)

            self._optimizer.zero_grad()
            loss = self._network.loss_function(states, actions)
            loss.backward()
            self._optimizer.step()

    def error(self, state0, action0):
        return self._network.error(state0, action0)

    def reward_sample(self, memory, indices):
        sample = memory.sample(indices)

        states = sample.state.to(self._device)
        actions = sample.action.to(self._device)

        return self.reward(states, actions)

    def reward(self, state0, action0):
        with torch.no_grad():
            reward = self.error(state0, action0).unsqueeze(1)
        return reward * self._eta

    def update_state_average(self, state, action):
        self._network.update_state_average(state, action)
