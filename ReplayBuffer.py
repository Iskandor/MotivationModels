import random
import torch


class Transition:
    def __init__(self, state0, action0, state1, reward, done):
        self.state0 = state0
        self.action0 = action0
        self.state1 = state1
        self.reward = reward
        self.done = done


class ReplayBuffer:

    def __init__(self, size):
        self._size = size
        self._buffer = []

    def add(self, item):
        if len(self._buffer) > self._size:
            self._buffer.pop(0)
        self._buffer.append(item)

    def sample(self, sample_size):
        if len(self._buffer) > sample_size:
            sample = random.sample(self._buffer, k=sample_size)
        else:
            sample = []

        return sample
