import itertools
import random
from collections import namedtuple, deque
from operator import itemgetter

import torch
import numpy as np

MDP_Transition = namedtuple('MDP_Transition', ('state', 'action', 'next_state', 'reward', 'mask'))
MDP_DOP_Transition = namedtuple('MDP_DOP_Transition', ('state', 'action', 'arbiter_action', 'next_state', 'reward', 'mask'))
M2_Transition = namedtuple('M2_Transition', ('state', 'action', 'next_state', 'gate_state', 'weight', 'next_gate_state', 'reward', 'mask'))
PPO_Transition = namedtuple('PPO_Transition', ('state', 'value', 'action', 'prob', 'next_state', 'reward', 'mask'))


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

        self.memory['state'][self.index] = state
        self.memory['action'][self.index] = action
        self.memory['next_state'][self.index] = next_state
        self.memory['reward'][self.index] = reward
        self.memory['mask'][self.index] = 1 - mask

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


class DOPReplayBuffer(ReplayBuffer):
    def dynamic_memory_init(self, state, action, arbiter_action, reward):
        shape = (self.capacity,) + tuple(state.shape[1:])
        self.memory['state'] = torch.zeros(shape)
        shape = (self.capacity,) + tuple(action.shape[1:])
        self.memory['action'] = torch.zeros(shape)
        shape = (self.capacity,) + tuple(arbiter_action.shape[1:])
        self.memory['arbiter_action'] = torch.zeros(shape)
        shape = (self.capacity,) + tuple(state.shape[1:])
        self.memory['next_state'] = torch.zeros(shape)
        shape = (self.capacity,) + tuple(reward.shape[1:])
        self.memory['reward'] = torch.zeros(shape)
        shape = (self.capacity, 1)
        self.memory['mask'] = torch.zeros(shape)

    def add(self, state, action, arbiter_action, next_state, reward, mask):
        if len(self.memory) == 0:
            self.dynamic_memory_init(state, action, arbiter_action, reward)

        self.memory['state'][self.index] = state
        self.memory['action'][self.index] = action
        self.memory['arbiter_action'][self.index] = arbiter_action
        self.memory['next_state'][self.index] = next_state
        self.memory['reward'][self.index] = reward
        self.memory['mask'][self.index] = 1 - mask

        self.index += 1
        if self.capacity_index < self.capacity:
            self.capacity_index += 1
        if self.index == self.capacity:
            self.index = 0

    def sample(self, indices):
        state = self.memory['state'][indices]
        action = self.memory['action'][indices]
        arbiter_action = self.memory['arbiter_action'][indices]
        next_state = self.memory['next_state'][indices]
        reward = self.memory['reward'][indices]
        mask = self.memory['mask'][indices]

        return MDP_DOP_Transition(state, action, arbiter_action, next_state, reward, mask)

    def sample_batches(self, indices):
        state = self.memory['state'][indices].unsqueeze(0)
        action = self.memory['action'][indices].unsqueeze(0)
        arbiter_action = self.memory['arbiter_action'][indices].unsqueeze(0)
        next_state = self.memory['next_state'][indices].unsqueeze(0)
        reward = self.memory['reward'][indices].unsqueeze(0)
        mask = self.memory['mask'][indices].unsqueeze(0)

        return MDP_DOP_Transition(state, action, arbiter_action, next_state, reward, mask), 1


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


class PPOTrajectoryBuffer2(object):
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
        self.memory['mask'][index] = 1 - mask
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
                self.memory[i].append(PPO_Transition(state[i], value[i], action[i], prob[i], next_state[i], reward[i], 1 - mask[i]))
        else:
            self.index += 1
            self.memory[0].append(PPO_Transition(state.squeeze(0), value.squeeze(0), action.squeeze(0), prob.squeeze(0), next_state.squeeze(0), reward.squeeze(0), 1 - mask))

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
