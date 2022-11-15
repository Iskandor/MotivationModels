import time

import torch


class ForwardModelMotivation:
    def __init__(self, network, lr, eta=1, variant='A', device='cpu'):
        self._network = network
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        self._eta = eta
        self._variant = variant
        self._device = device

    def train(self, memory, indices):
        if indices:
            start = time.time()
            sample, size = memory.sample_batches(indices)

            for i in range(size):
                states = sample.state[i].to(self._device)
                next_states = sample.next_state[i].to(self._device)
                actions = sample.action[i].to(self._device)

                self._optimizer.zero_grad()
                loss = self._network.loss_function(states, actions, next_states)
                loss.backward()
                self._optimizer.step()

            end = time.time()
            # print("Forward model motivation training time {0:.2f}s".format(end - start))

    def error(self, state0, action, state1):
        return self._network.error(state0, action, state1)

    def reward_sample(self, memory, indices):
        sample = memory.sample(indices)

        states = sample.state
        next_states = sample.next_state
        actions = sample.action

        return self.reward(states, actions, next_states)

    def reward(self, state0=None, action=None, state1=None, error=None):
        reward = 0
        if error is None:
            error = self.error(state0, action, state1)

        if self._variant == 'A':
            reward = torch.tanh(error)

        return reward * self._eta
