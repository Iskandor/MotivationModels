import torch


class ForwardModelMotivation:
    def __init__(self, network, lr, eta=1, variant='A', window=1, memory_buffer=None, sample_size=0):
        self._network = network
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        self._memory = memory_buffer
        self._sample_size = sample_size
        self._eta = eta
        self._variant = variant

        if self._variant == 'B':
            self._window = window
            self._window_index = 0
            self._error_buffer = torch.zeros(window, dtype=torch.float32)

    def train(self, state0, action, state1):
        if self._variant == 'B':
            self._error_buffer[self._window_index] = self.error(state0, action, state1)
            self._window_index += 1
            if self._window_index == self._window:
                self._window_index = 0

        if self._memory is not None:
            if len(self._memory) > self._sample_size:
                sample = self._memory.sample(self._sample_size)

                states = torch.stack(sample.state)
                next_states = torch.stack(sample.next_state)
                actions = torch.stack(sample.action)

                self._optimizer.zero_grad()
                loss = self._network.loss_function(states, actions, next_states)
                loss.backward()
                self._optimizer.step()
        else:
            self._optimizer.zero_grad()
            loss = self._network.loss_function(state0, action, state1)
            loss.backward()
            self._optimizer.step()

    def error(self, state0, action, state1):
        with torch.no_grad():
            dim = action.ndim - 1
            prediction = self._network(state0, action)
            error = torch.mean(torch.pow(prediction.flatten(dim) - state1.flatten(dim), 2), dim=dim).unsqueeze(dim)

        return error

    def mean_error(self):
        return self._error_buffer.mean()

    def reward(self, state0=None, action=None, state1=None, error=None):
        reward = 0
        if self._variant == 'A':
            if error is None:
                reward = torch.tanh(self.error(state0, action, state1))
            else:
                reward = torch.tanh(error)

        if self._variant == 'B':
            reward = torch.tanh(self.mean_error())

        return reward * self._eta

    def save(self, path):
        torch.save(self._network.state_dict(), path + '_fm.pth')

    def load(self, path):
        self._network.load_state_dict(torch.load(path + '_fm.pth'))
