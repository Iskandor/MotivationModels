import torch


class RNDMotivation:
    def __init__(self, network, lr, eta=1, memory_buffer=None, sample_size=0, device='cpu'):
        self._network = network
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        self._memory = memory_buffer
        self._sample_size = sample_size
        self._eta = eta
        self._device = device

    def train(self, indices):
        if indices:
            sample = self._memory.sample(indices)

            states = torch.stack(sample.state).squeeze(1)

            self._optimizer.zero_grad()
            loss = self._network.loss_function(states)
            loss.backward()
            self._optimizer.step()

    def error(self, state0):
        return self._network.error(state0)

    def reward_sample(self, indices):
        sample = self._memory.sample(indices)

        states = torch.stack(sample.state).squeeze(1)

        return self.reward(states)

    def reward(self, state0):
        reward = torch.tanh(self.error(state0)).unsqueeze(1)
        return reward * self._eta


class QRNDMotivation:
    def __init__(self, network, lr, eta=1, memory_buffer=None, sample_size=0, device='cpu'):
        self._network = network
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        self._memory = memory_buffer
        self._sample_size = sample_size
        self._eta = eta
        self._device = device

    def train(self, indices):
        if indices:
            sample = self._memory.sample(indices)

            states = torch.stack(sample.state).squeeze(1)
            actions = torch.stack(sample.action).squeeze(1)

            self._optimizer.zero_grad()
            loss = self._network.loss_function(states, actions)
            loss.backward()
            self._optimizer.step()

    def error(self, state0, action0):
        return self._network.error(state0, action0)

    def reward_sample(self, indices):
        sample = self._memory.sample(indices)

        states = torch.stack(sample.state).squeeze(1)
        actions = torch.stack(sample.action).squeeze(1)

        return self.reward(states, actions)

    def reward(self, state0, action0):
        reward = torch.tanh(self.error(state0, action0)).unsqueeze(1)
        return reward * self._eta


class DOPMotivation:
    def __init__(self, network, lr, eta=1, memory_buffer=None, sample_size=0, device='cpu'):
        self._network = network
        self._motivator_optimizer = torch.optim.Adam(self._network.motivator.parameters(), lr=lr)
        self._generator_optimizer = [torch.optim.Adam(g.parameters(), lr=lr) for g in self._network.generator]
        self._memory = memory_buffer
        self._sample_size = sample_size
        self._eta = eta
        self._device = device

    def train(self, indices):
        if indices:
            sample = self._memory.sample(indices)

            states = torch.stack(sample.state).squeeze(1)
            actions = torch.stack(sample.action).squeeze(1)
            noises = torch.stack(sample.noise).squeeze(1)
            index = torch.stack(sample.index).squeeze(1)

            self._motivator_optimizer.zero_grad()
            loss = self._network.motivator_loss_function(states, actions)
            loss.backward()
            self._motivator_optimizer.step()

            for i in index.tolist():
                self._generator_optimizer[i].zero_grad()
                loss = self._network.generator_loss_function(states, actions - noises, i)
                loss.backward()
                self._generator_optimizer[i].step()

    def error(self, state0, action0):
        return self._network.error(state0, action0)

    def reward_sample(self, indices):
        sample = self._memory.sample(indices)

        states = torch.stack(sample.state).squeeze(1)
        actions = torch.stack(sample.action).squeeze(1)

        return self.reward(states, actions)

    def reward(self, state0, action0):
        reward = torch.tanh(self.error(state0, action0)).unsqueeze(1)
        return reward * self._eta
