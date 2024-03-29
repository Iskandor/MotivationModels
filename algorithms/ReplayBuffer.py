import itertools
import random
from collections import namedtuple, deque
from operator import itemgetter

import torch
import numpy as np

MDP_Transition = namedtuple('MDP_Transition', ('state', 'action', 'next_state', 'reward', 'mask'))
M2_Transition = namedtuple('M2_Transition', ('state', 'action', 'next_state', 'gate_state', 'weight', 'next_gate_state', 'reward', 'mask'))


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.memory = {}
        self.index = 0
        self.capacity_index = 0
        self.capacity = capacity

    def __len__(self):
        return self.capacity_index

    def indices(self, sample_size):
        ind = None
        if len(self) > sample_size:
            ind = random.sample(range(0, len(self)), sample_size)
        return ind


class ExperienceReplayBuffer(ReplayBuffer):
    def dynamic_memory_init(self, state, action, reward):
        shape = (self.capacity,) + tuple(state.shape[1:])
        self.memory['state'] = torch.zeros(shape)
        shape = (self.capacity,) + tuple(action.shape[1:])
        self.memory['action'] = torch.zeros(shape)
        shape = (self.capacity,) + tuple(state.shape[1:])
        self.memory['next_state'] = torch.zeros(shape)
        shape = (self.capacity,) + tuple(reward.shape[1:])
        self.memory['reward'] = torch.zeros(shape)
        shape = (self.capacity, 1)
        self.memory['mask'] = torch.zeros(shape)

    def add(self, state, action, next_state, reward, mask):
        if len(self.memory) == 0:
            self.dynamic_memory_init(state, action, reward)

        self.memory['state'][self.index] = state.cpu()
        self.memory['action'][self.index] = action.cpu()
        self.memory['next_state'][self.index] = next_state.cpu()
        self.memory['reward'][self.index] = reward.cpu()
        self.memory['mask'][self.index] = 1 - mask.cpu()

        self.index += 1
        if self.capacity_index < self.capacity:
            self.capacity_index += 1
        if self.index == self.capacity:
            self.index = 0

    def sample(self, indices):
        state = self.memory['state'][indices]
        action = self.memory['action'][indices]
        next_state = self.memory['next_state'][indices]
        reward = self.memory['reward'][indices]
        mask = self.memory['mask'][indices]

        return MDP_Transition(state, action, next_state, reward, mask)

    def sample_batches(self, indices):
        state = self.memory['state'][indices].unsqueeze(0)
        action = self.memory['action'][indices].unsqueeze(0)
        next_state = self.memory['next_state'][indices].unsqueeze(0)
        reward = self.memory['reward'][indices].unsqueeze(0)
        mask = self.memory['mask'][indices].unsqueeze(0)

        return MDP_Transition(state, action, next_state, reward, mask), 1


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


class MDPTrajectoryBuffer(object):
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

    def dynamic_memory_init(self, state, action, reward):
        shape = (self.capacity // self.n_env, self.n_env,) + tuple(state.shape[1:])
        self.memory['state'] = torch.zeros(shape)
        shape = (self.capacity // self.n_env, self.n_env,) + tuple(action.shape[1:])
        self.memory['action'] = torch.zeros(shape)
        shape = (self.capacity // self.n_env, self.n_env,) + tuple(state.shape[1:])
        self.memory['next_state'] = torch.zeros(shape)
        shape = (self.capacity // self.n_env, self.n_env,) + tuple(reward.shape[1:])
        self.memory['reward'] = torch.zeros(shape)
        shape = (self.capacity // self.n_env, self.n_env, 1)
        self.memory['mask'] = torch.zeros(shape)

    def add(self, state, action, next_state, reward, mask):

        if len(self.memory) == 0:
            self.dynamic_memory_init(state, action, reward)

        index = self.index // self.n_env
        self.memory['state'][index] = state
        self.memory['action'][index] = action
        self.memory['next_state'][index] = next_state
        self.memory['reward'][index] = reward
        self.memory['mask'][index] = 1 - mask
        self.index += self.n_env

    def sample(self, indices, reshape_to_batch=True):
        if reshape_to_batch:
            result = MDP_Transition(
                self.memory['state'].reshape(-1, *self.memory['state'].shape[2:]),
                self.memory['action'].reshape(-1, *self.memory['action'].shape[2:]),
                self.memory['next_state'].reshape(-1, *self.memory['next_state'].shape[2:]),
                self.memory['reward'].reshape(-1, *self.memory['reward'].shape[2:]),
                self.memory['mask'].reshape(-1, *self.memory['mask'].shape[2:]))
        else:
            result = MDP_Transition(
                self.memory['state'],
                self.memory['action'],
                self.memory['next_state'],
                self.memory['reward'],
                self.memory['mask'])

        return result

    def sample_batches(self, indices):
        batch = MDP_Transition(
            self.memory['state'].reshape(-1, self.batch_size, *self.memory['state'].shape[2:]),
            self.memory['action'].reshape(-1, self.batch_size, *self.memory['action'].shape[2:]),
            self.memory['next_state'].reshape(-1, self.batch_size, *self.memory['next_state'].shape[2:]),
            self.memory['reward'].reshape(-1, self.batch_size, *self.memory['reward'].shape[2:]),
            self.memory['mask'].reshape(-1, self.batch_size, *self.memory['mask'].shape[2:]))
        return batch, self.capacity // self.batch_size

    def clear(self):
        self.index = 0


class GenericBuffer(object):
    def __init__(self, capacity, batch_size, n_env=1):
        self.keys = []
        self.n_env = n_env
        self.memory = {}
        self.index = 0
        self.capacity = capacity
        self.batch_size = batch_size
        self.transition = None

    def add(self, **kwargs):
        raise NotImplementedError

    def clear(self):
        pass


class GenericTrajectoryBuffer(GenericBuffer):
    def __len__(self):
        return self.index * self.n_env

    def memory_init(self, key, shape):
        steps_per_env = self.capacity // self.n_env
        shape = (steps_per_env, self.n_env,) + shape
        self.memory[key] = torch.zeros(shape)

    def add(self, **kwargs):
        if len(self.memory) == 0:
            for key in kwargs:
                self.keys.append(key)
                self.memory_init(key, tuple(kwargs[key].shape[1:]))
            self.transition = namedtuple('transition', self.keys)

        for key in kwargs:
            self.memory[key][self.index] = kwargs[key]

        self.index += 1

    def indices(self):
        ind = None
        if len(self) == self.capacity:
            ind = range(0, self.capacity)
        return ind

    def sample(self, indices, reshape_to_batch=True):
        if reshape_to_batch:
            values = [self.memory[k].reshape(-1, *self.memory[k].shape[2:]) for k in self.keys]
            result = self.transition(*values)
        else:
            values = [self.memory[k] for k in self.keys]
            result = self.transition(*values)

        return result

    def sample_batches(self, indices, batch_size=0):
        if batch_size == 0:
            batch_size = self.batch_size

        values = [self.memory[k].reshape(-1, batch_size, *self.memory[k].shape[2:]) for k in self.keys]
        batch = self.transition(*values)
        return batch, self.capacity // batch_size

    def clear(self):
        self.index = 0


class GenericReplayBuffer(GenericBuffer):
    def __init__(self, capacity, batch_size, n_env=1):
        super().__init__(capacity, batch_size, n_env)
        self.capacity_index = 0

    def __len__(self):
        return self.index

    def indices(self, sample_size):
        ind = None
        if len(self) > sample_size:
            ind = random.sample(range(0, len(self)), sample_size)
        return ind

    def memory_init(self, key, shape):
        self.memory[key] = torch.zeros(shape)

    def add(self, **kwargs):
        if len(self.memory) == 0:
            for key in kwargs:
                self.keys.append(key)
                self.memory_init(key, (self.capacity,) + tuple(kwargs[key].shape[1:]))
            self.transition = namedtuple('transition', self.keys)

        for key in kwargs:
            self.memory[key][self.index:self.index + self.n_env] = kwargs[key][:]

        self.index += self.n_env
        if self.capacity_index < self.capacity:
            self.capacity_index += self.n_env
        if self.index == self.capacity:
            self.index = 0

    def sample(self, indices):
        values = [self.memory[k][indices] for k in self.keys]
        result = self.transition(*values)

        return result

    def sample_batches(self, indices, batch_size=0):
        if batch_size == 0:
            batch_size = self.batch_size

        original_batch = len(indices)

        values = []

        for k in self.keys:
            v = self.memory[k][indices]
            values.append(v.reshape(-1, batch_size, *v.shape[1:]))

        batch = self.transition(*values)
        return batch, original_batch // batch_size


class GenericAsyncTrajectoryBuffer(object):
    def __init__(self, capacity, batch_size, n_env=1):
        self.keys = []
        self.n_env = n_env
        self.memory = {}
        self.index = 0
        self.capacity = capacity
        self.batch_size = batch_size
        self.transition = None

    def __len__(self):
        return self.index

    def indices(self):
        ind = None
        if len(self) >= self.capacity:
            ind = range(0, self.capacity)
        return ind

    def memory_init(self, key):
        self.memory[key] = [[] for _ in range(self.n_env)]

    def add(self, indices, **kwargs):
        if len(self.keys) == 0:
            for key in kwargs:
                self.keys.append(key)
                self.memory_init(key)
            self.transition = namedtuple('transition', self.keys)

        for i in indices:
            for key in kwargs:
                self.memory[key][i].append(kwargs[key][i])

        self.index += len(indices)

    def extract_value(self, key):
        v = []
        for i in range(self.n_env):
            if len(self.memory[key][i]) > 0:
                v.append(torch.stack(self.memory[key][i]))
        return torch.cat(v, dim=0)

    def sample(self, indices, reshape_to_batch=True):
        if reshape_to_batch:
            values = []

            for k in self.keys:
                v = self.extract_value(k)
                values.append(v.reshape(-1, 1, *v.shape[2:]))

        else:
            values = []

            for k in self.keys:
                values.append(self.extract_value(k).unsqueeze(1))

        result = self.transition(*values)

        return result

    def sample_batches(self, indices, batch_size=0):
        if batch_size == 0:
            batch_size = self.batch_size

        values = []

        for k in self.keys:
            v = self.extract_value(k)
            values.append(v.reshape(-1, batch_size, *v.shape[2:]))

        batch = self.transition(*values)
        return batch, self.capacity // batch_size

    def clear(self):
        self.index = 0
        for key in self.keys:
            self.memory_init(key)


if __name__ == '__main__':
    buffer = GenericReplayBuffer(100, 10, 2)

    for i in range(100):
        buffer.add(data=torch.rand(2, 4))
        indices = buffer.indices(10)
        if indices is not None:
            sample = buffer.sample_batches(indices, 10)
    pass