import numpy as np
from etaprogress.progress import ProgressBar


class RunningAverage:
    def __init__(self):
        self._cma = 0
        self._n = 0

    def update(self, value):
        self._cma = self._cma + (value - self._cma) / (self._n + 1)
        self._n += 1

    def value(self):
        return self._cma


class RunningAverageWindow:
    def __init__(self, window=1, size=1):
        self._cma = np.zeros((window, size))
        self._n = 0
        self._window = window

    def update(self, value):
        self._cma[self._n] = value
        self._n += 1
        if self._n == self._window:
            self._n = 0

    def value(self):
        return self._cma.mean(axis=0)


class StepCounter:
    def __init__(self, limit):
        self.limit = limit
        self.steps = 0
        self.bar = ProgressBar(limit, max_width=40)

    def update(self, value):
        self.steps += value
        self.bar.numerator = self.steps

    def print(self):
        print(self.bar)

    def running(self):
        return self.steps < self.limit
