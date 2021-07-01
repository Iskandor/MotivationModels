import itertools
import random
from collections import namedtuple, deque
from operator import itemgetter

import torch
import numpy as np

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
        self.memory = {}
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

    def dynamic_memory_init(self, state, action, prob):
        shape = (self.capacity // self.n_env, self.n_env,) + tuple(state.shape[1:])
        self.memory['state'] = torch.zeros(shape)
        shape = (self.capacity // self.n_env, self.n_env, 1)
        self.memory['value'] = torch.zeros(shape)
        shape = (self.capacity // self.n_env, self.n_env,) + tuple(action.shape[1:])
        self.memory['action'] = torch.zeros(shape)
        shape = (self.capacity // self.n_env, self.n_env,) + tuple(prob.shape[1:])
        self.memory['prob'] = torch.zeros(shape)
        shape = (self.capacity // self.n_env, self.n_env,) + tuple(state.shape[1:])
        self.memory['next_state'] = torch.zeros(shape)
        shape = (self.capacity // self.n_env, self.n_env, 1)
        self.memory['reward'] = torch.zeros(shape)
        shape = (self.capacity // self.n_env, self.n_env, 1)
        self.memory['mask'] = torch.zeros(shape)

    def add(self, state, value, action, prob, next_state, reward, mask):

        if len(self.memory) == 0:
            self.dynamic_memory_init(state, action, prob)

        index = self.index // self.n_env
        self.memory['state'][index] = state
        self.memory['value'][index] = value
        self.memory['action'][index] = action
        self.memory['prob'][index] = prob
        self.memory['next_state'][index] = next_state
        self.memory['reward'][index] = reward
        self.memory['mask'][index] = mask
        self.index += self.n_env

    def sample(self, indices):
        return PPO_Transition(
            self.memory['state'],
            self.memory['value'],
            self.memory['action'],
            self.memory['prob'],
            self.memory['next_state'],
            self.memory['reward'],
            self.memory['mask'])

    def sample_batches(self, indices):
        transitions = []
        for i in range(self.n_env):
            transitions += self.memory[i]
        batch = list(PPO_Transition(*zip(*transitions[x:x + self.batch_size])) for x in range(0, self.capacity, self.batch_size))

        return batch

    def clear(self):
        self.index = 0
