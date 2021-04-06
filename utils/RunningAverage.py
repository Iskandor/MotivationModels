import numpy as np


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
    def __init__(self, window=1):
        self._cma = np.zeros(window)
        self._n = 0
        self._window = window

    def update(self, value):
        self._cma[self._n] = value
        self._n += 1
        if self._n == self._window:
            self._n = 0

    def value(self):
        return self._cma.mean()
