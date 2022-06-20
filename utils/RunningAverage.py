import numpy as np
import torch
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
        if self.steps > self.limit:
            self.steps = self.limit
        self.bar.numerator = self.steps

    def print(self):
        print(self.bar)

    def running(self):
        return self.steps < self.limit


class RunningStatsSimple:
    def __init__(self, shape, device):
        self.count = 1
        self.eps = 0.0000001
        self.mean = torch.zeros(shape, device=device)
        self.var = 0.01 * torch.ones(shape, device=device)
        self.std = (self.var ** 0.5) + self.eps

    def update(self, x):
        self.count += 1

        mean = self.mean + (x.mean(axis=0) - self.mean) / self.count
        var = self.var + ((x - self.mean) * (x - mean)).mean(axis=0)

        self.mean = mean
        self.var = var

        self.std = ((self.var / self.count) ** 0.5) + self.eps


class RunningStats:
    def __init__(self, shape, device, n=1):
        self.n = n
        if n > 1:
            shape = (n,) + shape
            self.count = torch.ones((n, 1), device=device)
        else:
            self.count = 1
        self.eps = 0.0000001
        self.max = torch.zeros(shape, device=device)
        self.sum = torch.zeros(shape, device=device)
        self.mean = torch.zeros(shape, device=device)
        self.var = 0.01 * torch.ones(shape, device=device)
        self.std = (self.var ** 0.5) + self.eps

    def update(self, x, reduction='mean'):
        self.count += 1

        mean = None
        var = None
        max = torch.maximum(self.max, x)

        if reduction == 'mean':
            self.sum += x.mean(axis=0)
            mean = self.mean + (x.mean(axis=0) - self.mean) / self.count
            var = self.var + ((x - self.mean) * (x - mean)).mean(axis=0)
        if reduction == 'none':
            self.sum += x
            mean = self.mean + (x - self.mean) / self.count
            var = self.var + ((x - self.mean) * (x - mean))

        self.max = max
        self.mean = mean
        self.var = var

        self.std = ((self.var / self.count) ** 0.5) + self.eps

    def reset(self, i):
        if self.n > 1:
            self.max[i].fill_(0)
            self.sum[i].fill_(0)
            self.mean[i].fill_(0)
            self.var[i].fill_(0.01)
            self.count[i] = 1
        else:
            self.max.fill_(0)
            self.sum.fill_(0)
            self.mean.fill_(0)
            self.var.fill_(0.01)
            self.count = 1
