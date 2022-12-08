from collections import namedtuple

import torch

from utils.RunningAverage import RunningStats


class GenericCollector:
    def __init__(self):
        self.keys = []
        self._n_env = 0
        self._buffer = {}

        self._simple_stats = namedtuple('simple_stats', ['step', 'max', 'sum', 'mean', 'std'])

    def init(self, n_env, **kwargs):
        self._n_env = n_env
        for k in kwargs:
            self.add(k, kwargs[k])

    def add(self, key, shape):
        self.keys.append(key)
        self._buffer[key] = RunningStats(shape, 'cpu', n=self._n_env)

    def update(self, **kwargs):
        for k in kwargs:
            if k in self._buffer:
                self._buffer[k].update(kwargs[k], reduction='none')

    def reset(self, indices):
        result = {}

        for k in self._buffer:
            result[k] = []
            for i in indices:
                result[k].append(self._evaluate(k, i))
                self._buffer[k].reset(i)
            result[k] = self._simple_stats(*tuple(map(list, zip(*result[k]))))

        return result

    def _evaluate(self, key, index):
        return [self._buffer[key].count[index].item() - 1, self._buffer[key].max[index].item(), self._buffer[key].sum[index].item(), self._buffer[key].mean[index].item(), self._buffer[key].std[index].item()]

    def clear(self):
        self.keys.clear()
        self._n_env = 0
        self._buffer = {}
