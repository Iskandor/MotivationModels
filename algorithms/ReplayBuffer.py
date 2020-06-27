import random
from collections import namedtuple, deque

import torch

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'mask'))


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, state, action, next_state, reward, mask):
        if mask:
            self.memory.append(Transition(state, action, next_state, torch.tensor([reward], dtype=torch.float32), torch.tensor([0], dtype=torch.float32)))
        else:
            self.memory.append(Transition(state, action, next_state, torch.tensor([reward], dtype=torch.float32), torch.tensor([1], dtype=torch.float32)))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)