import torch

from algorithms.DDPG import DDPG


class M2SMotivation:
    def __init__(self, network, lr, eta, memory, sample_size, training_length, mean=0.5, squeeze=1.0):
        self.network = network
        self._optimizer = torch.optim.Adam(self.network.forward_model.parameters(), lr=lr)
        self.eta = eta
        self._memory = memory
        self._sample_size = sample_size
        self._training_length = training_length
        self._training_step = torch.tensor([0.], dtype=torch.float32)
        self._xi_scale = 12
        self._xi = torch.tensor([0.], dtype=torch.float32)
        self._xi_mean = mean
        self._xi_squeeze = squeeze

    def train(self, indices):
        self.update_xi()

        if indices:
            sample = self._memory.sample(indices)

            states = torch.stack(sample.state).squeeze(1)
            next_states = torch.stack(sample.next_state).squeeze(1)
            actions = torch.stack(sample.action).squeeze(1)

            self._optimizer.zero_grad()
            loss = self.network.forward_model.loss_function(states, actions, next_states)
            loss.backward()
            self._optimizer.step()

    def reward_sample(self, indices):
        sample = self._memory.sample(indices)

        states = torch.stack(sample.state).squeeze(1)
        next_states = torch.stack(sample.next_state).squeeze(1)
        actions = torch.stack(sample.action).squeeze(1)

        return self.reward(states, actions, next_states)

    def reward(self, state0, action, state1):
        weights = self.weight(state0)
        with torch.no_grad():
            error = self.network.forward_model.error(state0, action, state1)
            rewards = torch.cat([self._curious_reward(error), self._familiar_reward(error)], dim=1)

            reward = rewards * weights
            reward = reward.sum(dim=1).unsqueeze(1)

        return reward * self.eta

    def error(self, state0, action, state1):
        return self.network.forward_model.error(state0, action, state1)

    def weight(self, gate_state):
        with torch.no_grad():
            weight = torch.tensor([[1 - self._xi, self._xi]], dtype=torch.float32)
            weight = weight.repeat(gate_state.shape[0], 1)
            return weight

    def update_xi(self):
        self._xi = torch.sigmoid(self._training_step / self._training_length * self._xi_scale - self._xi_scale * self._xi_mean)
        self._training_step += 1

    @staticmethod
    def _curious_reward(error):
        return torch.tanh(error)

    @staticmethod
    def _familiar_reward(error):
        return torch.tanh(1 / error)
