import itertools

import numpy as np
import torch

from analytic.GenericCollector import GenericCollector


class ResultCollector:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print('Creating the object ResultCollector')
            cls._instance = super(ResultCollector, cls).__new__(cls)
            cls._instance.collector = GenericCollector()
            cls._instance.global_step = 0
            cls._instance.n_env = 0

            cls._instance.collector_values = {}
            cls._instance.global_values = {}
        return cls._instance

    def init(self, n_env, **kwargs):
        self.collector.init(n_env, **kwargs)
        self.n_env = n_env

        for k in self.collector.keys:
            self.collector_values[k] = []

    def add(self, **kwargs):
        for k in kwargs:
            if k not in self.collector_values:
                self.collector.add(k, kwargs[k])
                self.collector_values[k] = []

    def update(self, **kwargs):
        self.collector.update(**kwargs)

        for k in [item for item in kwargs.keys() if item not in self.collector.keys]:
            if k not in self.global_values:
                self.global_values[k] = {}
            if self.global_step not in self.global_values[k]:
                self.global_values[k][self.global_step] = []
            self.global_values[k][self.global_step].append(kwargs[k].cpu().item())

    def reset(self, indices):
        result = None
        if len(indices) > 0:
            result = self.collector.reset(indices)

            for k in self.collector.keys:
                self.collector_values[k].append((result[k].step, result[k].sum, result[k].max, result[k].mean, result[k].std))

        return result

    def end_step(self):
        self.global_step += self.n_env

    def finalize(self):
        data = {}

        for k in self.collector_values.keys():
            data[k] = self._finalize_value(self.collector_values[k], ['step', 'sum', 'max', 'mean', 'std'], mode='cumsum_step')

        for k in self.global_values.keys():
            data[k] = self._finalize_value(self.global_values[k], ['step', 'val'], mode='mean_step')

        return data

    def clear(self):
        self.collector.clear()
        self.global_step = 0
        self.n_env = 0

        for k in self.collector_values.keys():
            self.collector_values[k].clear()
        self.collector_values.clear()

        for k in self.global_values.keys():
            self.global_values[k].clear()
        self.global_values.clear()

    @staticmethod
    def _finalize_value(value, keys, mode):
        result = {}

        if mode == 'cumsum_step':
            l = tuple(map(list, zip(*value)))
            for i, k in enumerate(keys):
                result[k] = np.array(list(itertools.chain(*l[i])))

            result['step'] = np.cumsum(result['step'])

        if mode == 'mean_step':
            l = []
            for steps in value:
                val = sum(value[steps]) / len(value[steps])
                l.append((steps, val))

            l = tuple(map(list, zip(*l)))
            for i, k in enumerate(keys):
                result[k] = np.array(l[i])

        return result


if __name__ == '__main__':
    analytic = ResultCollector()

    analytic.init(8, ext_reward=(1,), int_reward=(1,), error=(1,), score=(1,))
    for i in range(10):
        analytic.update(ext_reward=torch.rand((8, 1)), int_reward=torch.rand((8, 1)), error=torch.rand((8, 1)),
                        score=torch.rand((8, 1)))
        analytic.end_step()
    analytic.reset([1, 3])
    for i in range(4):
        analytic.update(ext_reward=torch.rand((8, 1)), int_reward=torch.rand((8, 1)), error=torch.rand((8, 1)),
                        score=torch.rand((8, 1)))
        analytic.end_step()
    analytic.reset([0, 1, 2])
    analytic.finalize()
    pass
