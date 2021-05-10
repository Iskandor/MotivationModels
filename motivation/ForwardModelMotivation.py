import torch


class ForwardModelMotivation:
    def __init__(self, network, lr, eta=1, variant='A', memory_buffer=None, device='cpu'):
        self._network = network
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        self._memory = memory_buffer
        self._eta = eta
        self._variant = variant
        self._device = device

    def train(self, indices):
        if indices:
            sample_batch = self._memory.sample_batches(indices)

            for sample in sample_batch:
                states = torch.stack(sample.state).squeeze(1).to(self._device)
                next_states = torch.stack(sample.next_state).squeeze(1).to(self._device)
                actions = torch.stack(sample.action).squeeze(1).to(self._device)

                self._optimizer.zero_grad()
                loss = self._network.loss_function(states, actions, next_states)
                loss.backward()
                self._optimizer.step()

    def error(self, state0, action, state1):
        return self._network.error(state0, action, state1)

    def reward_sample(self, indices):
        sample = self._memory.sample(indices)

        states = torch.stack(sample.state).squeeze(1)
        next_states = torch.stack(sample.next_state).squeeze(1)
        actions = torch.stack(sample.action).squeeze(1)

        return self.reward(states, actions, next_states)

    def reward(self, state0=None, action=None, state1=None, error=None):
        reward = 0
        if error is None:
            error = self.error(state0, action, state1)

        if self._variant == 'A':
            reward = torch.tanh(error)

        return reward * self._eta
