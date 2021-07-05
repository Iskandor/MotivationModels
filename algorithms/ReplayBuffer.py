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

    def dynamic_memory_init(self, state, value, action, prob, reward):
        shape = (self.capacity // self.n_env, self.n_env,) + tuple(state.shape[1:])
        self.memory['state'] = torch.zeros(shape)
        shape = (self.capacity // self.n_env, self.n_env,) + tuple(value.shape[1:])
        self.memory['value'] = torch.zeros(shape)
        shape = (self.capacity // self.n_env, self.n_env,) + tuple(action.shape[1:])
        self.memory['action'] = torch.zeros(shape)
        shape = (self.capacity // self.n_env, self.n_env,) + tuple(prob.shape[1:])
        self.memory['prob'] = torch.zeros(shape)
        shape = (self.capacity // self.n_env, self.n_env,) + tuple(state.shape[1:])
        self.memory['next_state'] = torch.zeros(shape)
        shape = (self.capacity // self.n_env, self.n_env,) + tuple(reward.shape[1:])
        self.memory['reward'] = torch.zeros(shape)
        shape = (self.capacity // self.n_env, self.n_env, 1)
        self.memory['mask'] = torch.zeros(shape)

    def add(self, state, value, action, prob, next_state, reward, mask):

        if len(self.memory) == 0:
            self.dynamic_memory_init(state, value, action, prob, reward)

        index = self.index // self.n_env
        self.memory['state'][index] = state
        self.memory['value'][index] = value
        self.memory['action'][index] = action
        self.memory['prob'][index] = prob
        self.memory['next_state'][index] = next_state
        self.memory['reward'][index] = reward
        self.memory['mask'][index] = mask
        self.index += self.n_env

    def sample(self, indices, reshape_to_batch=True):
        if reshape_to_batch:
            result = PPO_Transition(
                self.memory['state'].reshape(-1, *self.memory['state'].shape[2:]),
                self.memory['value'].reshape(-1, *self.memory['value'].shape[2:]),
                self.memory['action'].reshape(-1, *self.memory['action'].shape[2:]),
                self.memory['prob'].reshape(-1, *self.memory['prob'].shape[2:]),
                self.memory['next_state'].reshape(-1, *self.memory['next_state'].shape[2:]),
                self.memory['reward'].reshape(-1, *self.memory['reward'].shape[2:]),
                self.memory['mask'].reshape(-1, *self.memory['mask'].shape[2:]))
        else:
            result = PPO_Transition(
                self.memory['state'],
                self.memory['value'],
                self.memory['action'],
                self.memory['prob'],
                self.memory['next_state'],
                self.memory['reward'],
                self.memory['mask'])

        return result

    def sample_batches(self, indices):
        batch = PPO_Transition(
            self.memory['state'].reshape(-1, self.batch_size, *self.memory['state'].shape[2:]),
            self.memory['value'].reshape(-1, self.batch_size, *self.memory['value'].shape[2:]),
            self.memory['action'].reshape(-1, self.batch_size, *self.memory['action'].shape[2:]),
            self.memory['prob'].reshape(-1, self.batch_size, *self.memory['prob'].shape[2:]),
            self.memory['next_state'].reshape(-1, self.batch_size, *self.memory['next_state'].shape[2:]),
            self.memory['reward'].reshape(-1, self.batch_size, *self.memory['reward'].shape[2:]),
            self.memory['mask'].reshape(-1, self.batch_size, *self.memory['mask'].shape[2:]))
        return batch, self.capacity // self.batch_size

    def clear(self):
        self.index = 0
