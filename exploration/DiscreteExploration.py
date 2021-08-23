import random
import numpy


class DiscreteExploration:
    def __init__(self, epsilon0, epsilon1=0, epochs=0):
        self.epsilon = epsilon0
        self._epsilon0 = epsilon0
        self._epsilon1 = epsilon1
        self._epochs = epochs

    def explore(self, action, env):
        if random.random() > self.epsilon:
            return action
        else:
            return env.action_space.sample()

    def update(self, step):
        if self._epsilon0 > self._epsilon1 and self.epsilon > self._epsilon1 and self._epochs > 0:
            self.epsilon = numpy.interp(step, [0, self._epochs], [self._epsilon0, self._epsilon1])
