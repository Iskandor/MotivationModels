import torch


class RNDMotivation:
    def __init__(self, network, lr, eta=1, memory_buffer=None, device='cpu'):
        self._network = network
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        self._memory = memory_buffer
        self._eta = eta
        self._device = device

    def train(self, indices):
        if indices:
            sample, size = self._memory.sample_batches(indices)

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

        states = sample.state

        return self.reward(states)

    def reward(self, state0):
        reward = self.error(state0).unsqueeze(1)
        return reward * self._eta

    def update_state_average(self, state):
        self._network.update_state_average(state)


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

            states = sample.state
            actions = sample.action

            self._optimizer.zero_grad()
            loss = self._network.loss_function(states, actions)
            loss.backward()
            self._optimizer.step()

    def error(self, state0, action0):
        return self._network.error(state0, action0)

    def reward_sample(self, memory, indices):
        sample = memory.sample(indices)

        states = sample.state
        actions = sample.action

        return self.reward(states, actions)

    def reward(self, state0, action0):
        reward = torch.tanh(self.error(state0, action0)).unsqueeze(1)
        return reward * self._eta

    def update_state_average(self, state):
        self._network.update_state_average(state)


class DOPSimpleMotivation:
    def __init__(self, network, lr, eta=1, memory_buffer=None, device='cpu'):
        self.network = network
        self._motivator_optimizer = torch.optim.Adam(self.network.motivator.parameters(), lr=lr)
        self._generator_optimizer = torch.optim.Adam(self.network.actor.parameters(), lr=0.0001)
        self._memory = memory_buffer
        self._eta = eta
        self._device = device

    def train(self, indices):
        if indices:
            sample = self._memory.sample(indices)

            states = sample.state
            actions = sample.action

            self._motivator_optimizer.zero_grad()
            loss = self.network.motivator_loss_function(states, actions)
            loss.backward()
            self._motivator_optimizer.step()

            self._generator_optimizer.zero_grad()
            loss = self.network.generator_loss_function(states)
            loss.backward()
            self._generator_optimizer.step()
            # print(loss)

    def error(self, state0, action0):
        return self.network.error(state0, action0)

    def reward_sample(self, memory, indices):
        sample = memory.sample(indices)

        states = sample.state
        actions = sample.action

        return self.reward(states, actions)

    def reward(self, state0, action0):
        reward = torch.tanh(self.error(state0, action0)).unsqueeze(1)
        return reward * self._eta


class DOPMotivation:
    def __init__(self, network, lr, eta=1, memory_buffer=None, sample_size=0, device='cpu'):
        self._network = network
        self._motivator_optimizer = torch.optim.Adam(self._network.motivator.parameters(), lr=lr)
        self._generator_optimizer = torch.optim.Adam(self._network.actor.parameters(), lr=lr)
        self._memory = memory_buffer
        self._sample_size = sample_size
        self._eta = eta
        self._device = device

    def train(self, indices):
        if indices:
            sample = self._memory.sample(indices)

            states = sample.state
            actions = sample.action

            self._motivator_optimizer.zero_grad()
            loss = self._network.motivator_loss_function(states, actions)
            loss.backward()
            self._motivator_optimizer.step()

            self._generator_optimizer.zero_grad()
            loss = self._network.generator_loss_function(states)
            loss.backward()
            self._generator_optimizer.step()

    def error(self, state0, action0):
        return self._network.error(state0, action0).detach()

    def reward_sample(self, indices):
        sample = self._memory.sample(indices)

        states = sample.state
        actions = sample.action

        return self.reward(states, actions)

    def reward(self, state0, action0):
        reward = torch.tanh(self.error(state0, action0)).unsqueeze(1)
        return reward * self._eta

    def update_state_average(self, state):
        self._network.motivator.update_state_average(state)
