import itertools
import random
from collections import namedtuple, deque
from operator import itemgetter

import torch

MDP_Transition = namedtuple('MDP_Transition', ('state', 'action', 'next_state', 'reward', 'mask'))
MDP_DOP_Transition = namedtuple('MDP_DOP_Transition', ('state', 'action', 'noise', 'index', 'next_state', 'reward', 'mask'))
M2_Transition = namedtuple('M2_Transition', ('state', 'action', 'next_state', 'gate_state', 'weight', 'next_gate_state', 'reward', 'mask'))
PPO_Transition = namedtuple('PPO_Transition', ('state', 'value', 'action', 'prob', 'next_state', 'reward', 'mask'))


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def __len__(self):
        return len(self.memory)

    def indices(self, sample_size):
        ind = None
        if len(self) > sample_size:
            ind = random.sample(range(0, len(self.memory)), sample_size)
        return ind


class ExperienceReplayBuffer(ReplayBuffer):
    def add(self, state, action, next_state, reward, mask):
        if mask:
            self.memory.append(MDP_Transition(state, action, next_state, torch.tensor([reward], dtype=torch.float32), torch.tensor([0], dtype=torch.float32)))
        else:
            self.memory.append(MDP_Transition(state, action, next_state, torch.tensor([reward], dtype=torch.float32), torch.tensor([1], dtype=torch.float32)))

    def sample(self, indices):
        transitions = list(itemgetter(*indices)(self.memory))
        batch = MDP_Transition(*zip(*transitions))

        return batch

    def sample_batches(self, indices):
        return [self.sample(indices)]


class DOPReplayBuffer(ReplayBuffer):
    def add(self, state, action, noise, index, next_state, reward, mask):
        if mask:
            self.memory.append(MDP_DOP_Transition(state, action, noise, index, next_state, torch.tensor([reward], dtype=torch.float32), torch.tensor([0], dtype=torch.float32)))
        else:
            self.memory.append(MDP_DOP_Transition(state, action, noise, index, next_state, torch.tensor([reward], dtype=torch.float32), torch.tensor([1], dtype=torch.float32)))

    def sample(self, indices):
        transitions = list(itemgetter(*indices)(self.memory))
        batch = MDP_DOP_Transition(*zip(*transitions))

        return batch

    def sample_batches(self, indices):
        return [self.sample(indices)]


class M2ReplayBuffer(ReplayBuffer):
    def add(self, state, action, next_state, gate_state, weight, next_gate_state, reward, mask):
        if mask:
            self.memory.append(M2_Transition(state, action, next_state, gate_state, weight, next_gate_state, torch.tensor([reward], dtype=torch.float32), torch.tensor([0], dtype=torch.float32)))
        else:
            self.memory.append(M2_Transition(state, action, next_state, gate_state, weight, next_gate_state, torch.tensor([reward], dtype=torch.float32), torch.tensor([1], dtype=torch.float32)))

    def sample(self, indices):
        transitions = list(itemgetter(*indices)(self.memory))
        batch = M2_Transition(*zip(*transitions))

        return batch

    def sample_batches(self, indices):
        return [self.sample(indices)]


class PPOTrajectoryBuffer(object):
    def __init__(self, capacity, batch_size, n_env=1):
        self.n_env = n_env
        self.memory = [[] for _ in range(self.n_env)]
        self.index = 0
        self.capacity = capacity
        self.batch_size = batch_size

    def __len__(self):
        return self.index

    def indices(self):
        ind = None
        if len(self) == self.capacity:
            ind = range(0, self.capacity)
        return ind

    def add(self, state, value, action, prob, next_state, reward, mask):
        if self.n_env > 1:
            self.index += self.n_env
            for i in range(self.n_env):
                self.memory[i].append(PPO_Transition(state[i], value[i], action[i], prob[i], next_state[i], reward[i], mask[i]))
        else:
            self.index += 1
            self.memory[0].append(PPO_Transition(state.squeeze(0), value.squeeze(0), action.squeeze(0), prob.squeeze(0), next_state.squeeze(0), torch.tensor([reward], dtype=torch.float32),
                                                 torch.tensor([mask], dtype=torch.float32)))

    def sample(self, indices):
        transitions = []
        for i in range(self.n_env):
            transitions += self.memory[i]
        batch = PPO_Transition(*zip(*transitions))

        return batch

    def sample_batches(self, indices):
        transitions = []
        for i in range(self.n_env):
            transitions += self.memory[i]
        batch = list(PPO_Transition(*zip(*transitions[x:x + self.batch_size])) for x in range(0, self.capacity, self.batch_size))

        return batch

    def clear(self):
        self.index = 0
        for i in range(self.n_env):
            del self.memory[i][:]
