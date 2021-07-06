import torch

from algorithms.DDPG import DDPG


class M2Motivation:
    def __init__(self, network, lr, gamma, tau, eta, memory, sample_size):
        self.network = network
        self._optimizer = torch.optim.Adam(self.network.forward_model.parameters(), lr=lr)
        self.eta = eta
        self._memory = memory
        self._sample_size = sample_size
        self._weight = torch.tensor([[0.9, 0.1]], dtype=torch.float32)

        self.gate_algorithm = DDPG(self.network.gate, lr, lr, gamma, tau, memory, 32)

    def train(self, indices):
        if indices:
            sample = self._memory.sample(indices)

            states = torch.stack(sample.state).squeeze(1)
            next_states = torch.stack(sample.next_state).squeeze(1)
            actions = torch.stack(sample.action).squeeze(1)
            # gate_state = torch.stack(sample.gate_state).squeeze(1)
            # next_gate_state = torch.stack(sample.next_gate_state).squeeze(1)
            # weights = torch.stack(sample.weight).squeeze(1)
            # rewards = torch.stack(sample.reward)
            # masks = torch.stack(sample.mask)

            self._optimizer.zero_grad()
            loss = self.network.forward_model.loss_function(states, actions, next_states)
            loss.backward()
            self._optimizer.step()

            # self.gate_algorithm.train(gate_state, weights, next_gate_state, rewards, masks)

    def reward_sample(self, indices):
        sample = self._memory.sample(indices)

        states = torch.stack(sample.state).squeeze(1)
        next_states = torch.stack(sample.next_state).squeeze(1)
        actions = torch.stack(sample.action).squeeze(1)
        gate_state = torch.stack(sample.gate_state).squeeze(1)
        weights = self.weight(gate_state)

        return self.reward(states, actions, weights, next_states)

    def reward(self, state0, action, weight, state1):
        with torch.no_grad():
            error = self.network.forward_model.error(state0, action, state1)
            rewards = torch.cat([self._curious_reward(error), self._familiar_reward(error)], dim=1)

            reward = rewards * weight
            reward = reward.sum(dim=1).unsqueeze(1)

        return reward * self.eta

    def weight(self, gate_state):
        with torch.no_grad():
            # weight = self.network.weight(gate_state)
            weight = self._weight.repeat(gate_state.shape[0], 1)
            return weight

    @staticmethod
    def _curious_reward(error):
        return torch.tanh(error)

    @staticmethod
    def _familiar_reward(error):
        return torch.tanh(1 / error)
