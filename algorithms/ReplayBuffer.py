import random
from collections import namedtuple, deque

import torch

MDP_Transition = namedtuple('MDP_Transition', ('state', 'action', 'next_state', 'reward', 'mask'))
Model_Transition = namedtuple('Model_Transition', ('state', 'action', 'next_state'))


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def __len__(self):
        return len(self.memory)


class ExperienceReplayBuffer(ReplayBuffer):
    def add(self, state, action, next_state, reward, mask):
        if mask:
            self.memory.append(MDP_Transition(state, action, next_state, torch.tensor([reward], dtype=torch.float32), torch.tensor([0], dtype=torch.float32)))
        else:
            self.memory.append(MDP_Transition(state, action, next_state, torch.tensor([reward], dtype=torch.float32), torch.tensor([1], dtype=torch.float32)))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = MDP_Transition(*zip(*transitions))
        return batch


class ModelReplayBuffer(ReplayBuffer):
    def add(self, state, action, next_state):
        self.memory.append(Model_Transition(state, action, next_state))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Model_Transition(*zip(*transitions))
        return batch
