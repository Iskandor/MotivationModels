import time

import torch

from utils.RunningAverage import RunningStats


class SNDMotivationFactory:
    @staticmethod
    def get_motivation(type, network, lr, eta=1, device='cpu'):
        if type == 'vanilla':
            result = SNDVMotivation(network, lr, eta, device)
        elif type == 'vinv':
            result = SINVMotivation(network, lr, eta, device)
        else:
            result = CNDMotivation(network, lr, eta, device)

        return result


class CNDMotivation:
    def __init__(self, network, lr, eta=1, device='cpu'):
        self._network = network
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        self._eta = eta
        self._device = device
        self.reward_stats = RunningStats(1, device)

    def train(self, memory, indices):
        if indices:
            start = time.time()
            sample, size = memory.sample_batches(indices)

            for i in range(size):
                states = sample.state[i].to(self._device)
                next_states = sample.next_state[i].to(self._device)

                self._optimizer.zero_grad()
                loss = self._network.loss_function(states, next_states)
                loss.backward()
                self._optimizer.step()

            end = time.time()
            print("CND motivation training time {0:.2f}s".format(end - start))

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


class SNDVMotivation(CNDMotivation):
    def train(self, memory, indices):
        if indices:
            start = time.time()
            sample_batch, size = memory.sample_batches(indices)
            sample = memory.sample(indices)
            states = sample.state

            for i in range(size):
                state_batch = sample_batch.state[i].to(self._device)

                self._optimizer.zero_grad()
                loss = self._network.loss_function(state_batch, states)
                loss.backward()
                self._optimizer.step()

            end = time.time()
            print("SNDV motivation training time {0:.2f}s".format(end - start))


class SINVMotivation(CNDMotivation):
    def train(self, memory, indices):
        if indices:
            start = time.time()
            sample, size = memory.sample_batches(indices)

            for i in range(size):
                states = sample.state[i].to(self._device)
                next_states = sample.next_state[i].to(self._device)
                actions = sample.action[i].to(self._device)

                self._optimizer.zero_grad()
                loss = self._network.loss_function(states, next_states, actions)
                loss.backward()
                self._optimizer.step()

            end = time.time()
            print("SINV motivation training time {0:.2f}s".format(end - start))
