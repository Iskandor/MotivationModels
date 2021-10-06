import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np


class BitFlipEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        print('BitFlip environment dimension: {0:d}'.format(kwargs['dimension']))
        self.dimension = kwargs['dimension']
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.dimension * 2,), dtype=np.int32)
        self.action_space = spaces.Discrete(self.dimension)
        self._state = None
        self._goal = None
        self._obs = None
        self._step = 0
        self._max_steps = pow(self.dimension, 2) * 2

    def step(self, action):
        self._step += 1
        reward = 0.
        done = self._step == self._max_steps

        self._state[action] = 1 - self._state[action]
        self._obs[action] = 1 - self._obs[action]

        if (self._state == self._goal).all():
            reward = 1.
            done = True

        obs = self._obs

        return obs, reward, done, {}

    def reset(self):
        self._step = 0
        self._state = np.random.randint(2, size=self.dimension)
        self._goal = np.random.randint(2, size=self.dimension)
        self._obs = np.concatenate([self._state, self._goal])
        return self._obs

    def render(self, mode='human'):
        print('{1:s} - {2:s}'.format(self._step, np.array2string(self._state), np.array2string(self._goal)))

    def close(self):
        pass
