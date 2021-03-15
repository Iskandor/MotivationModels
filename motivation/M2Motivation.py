import torch

from algorithms.DDPG2 import DDPG2


class M2Motivation:
    def __init__(self, network, lr, gamma, tau, eta, model_memory_buffer, gate_memory_buffer, sample_size):
        self.network = network
        self._optimizer = torch.optim.Adam(self.network.forward_model.parameters(), lr=lr)
        self.eta = eta
        self._model_memory = model_memory_buffer
        self._gate_memory = gate_memory_buffer
        self._sample_size = sample_size

        self.gate_algorithm = DDPG2(self.network, lr, lr, gamma, tau, self._gate_memory, 32)

    def train(self, state0, im0, action, weight, state1, im1, reward, done):
        self.gate_algorithm.train(im0, weight, im1, reward, done)

        if self._model_memory is not None:
            if len(self._model_memory) > self._sample_size:
                sample = self._model_memory.sample(self._sample_size)

                states = torch.stack(sample.state).squeeze(1)
                next_states = torch.stack(sample.next_state).squeeze(1)
                actions = torch.stack(sample.action).squeeze(1)

                self._optimizer.zero_grad()
                loss = self.network.forward_model.loss_function(states, actions, next_states)
                loss.backward()
                self._optimizer.step()
        else:
            self._optimizer.zero_grad()
            loss = self.network.forward_model.loss_function(state0, action, state1)
            loss.backward()
            self._optimizer.step()

    def reward(self, state0, action, weight, state1):
        with torch.no_grad():
            error = self.network.forward_model.error(state0, action, state1)
            rewards = torch.cat([self._curious_reward(error), self._familiar_reward(error)], dim=1)

            reward = rewards * weight
            reward = reward.sum(dim=1).unsqueeze(1)

        return reward * self.eta

    def weight(self, im):
        with torch.no_grad():
            weight = self.network.weight(im)
            return weight

    @staticmethod
    def _curious_reward(error):
        return torch.tanh(error)

    @staticmethod
    def _familiar_reward(error):
        return torch.tanh(1 / error)