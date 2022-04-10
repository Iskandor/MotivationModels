from collections import namedtuple

import torch

from utils.RunningAverage import RunningStats


class GenericCollector:
    def __init__(self, nenv):
        self._nenv = nenv
        self._buffer = {}

        self._simple_stats = namedtuple('simple_stats', ['step', 'max', 'sum', 'mean', 'std'])

    def init(self, **kwargs):
        for k in kwargs:
            self._buffer[k] = RunningStats(kwargs[k], 'cpu', n=self._nenv)

    def update(self, **kwargs):
        for k in kwargs:
            if k not in self._buffer:
                print('{0:s} update: Wrong buffer key "{1:s}"'.format(__file__, k))
            else:
                self._buffer[k].update(kwargs[k], reduction='none')

    def reset(self, indices):
        result = {}

        for k in self._buffer:
            result[k] = []
            for i in indices:
                result[k].append(self._evaluate(k, i))
                self._buffer[k].reset(i)
            result[k] = self._simple_stats(*[torch.tensor(l).unsqueeze(-1) for l in zip(*result[k])])

        return result

    def _evaluate(self, key, index):
        return self._buffer[key].count[index].item() - 1, self._buffer[key].max[index].item(), self._buffer[key].sum[index].item(), self._buffer[key].mean[index].item(), self._buffer[key].std[index].item()
