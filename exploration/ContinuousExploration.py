import numpy
import torch


class GaussianExploration:
    def __init__(self, sigma0, sigma1=0, epochs=0):
        self._sigma = sigma0
        self._sigma0 = sigma0
        self._sigma1 = sigma1
        self._epochs = epochs

    def explore(self, action):
        return action + torch.normal(0, self._sigma, action.size())

    def update(self, step):
        if self._sigma0 > self._sigma1 and self._sigma > self._sigma1 and self._epochs > 0:
            self._sigma = numpy.interp(step, [0, self._epochs], [self._sigma0, self._sigma1])


class OUExploration:
    def __init__(self, action_dimension, sigma0, sigma1=0, epochs=0, mu=0, theta=0.15, dt=1e-2):
        self._action_dimension = action_dimension
        self._dt = dt
        self._mu = mu
        self._theta = theta
        self._sigma = sigma0
        self._sigma0 = sigma0
        self._sigma1 = sigma1
        self._epochs = epochs
        self._state = numpy.ones(self._action_dimension) * self._mu

    def explore(self, action):
        return action + torch.tensor(self.noise(), dtype=torch.float32)

    def reset(self):
        self._state = numpy.ones(self._action_dimension) * self._mu

    def noise(self):
        x = self._state
        dx = self._theta * (self._mu - x) * self._dt + self._sigma * numpy.random.randn(len(x)) * numpy.sqrt(self._dt)
        self._state = x + dx
        return self._state
