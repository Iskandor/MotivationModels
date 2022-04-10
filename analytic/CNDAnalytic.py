import torch

from analytic.GenericCollector import GenericCollector


class CNDAnalytic(GenericCollector):
    _instance = None

    def __new__(cls, nenv):
        if cls._instance is None:
            print('Creating the object CNDAnalytic')
            cls._instance = super(CNDAnalytic, cls).__new__(cls)
        return cls._instance

    def __init__(self, nenv):
        super().__init__(nenv)

        self.ext_reward = []
        self.int_reward = []
        self.error = []
        self.score = []

    def reset(self, indices):
        result = None
        if len(indices) > 0:
            result = super().reset(indices)

            self.ext_reward.append((result['ext_reward'].step, result['ext_reward'].mean))
            self.int_reward.append((result['int_reward'].step, result['int_reward'].max, result['int_reward'].mean, result['int_reward'].std))
            self.error.append((result['error'].step, result['error'].max, result['error'].mean, result['error'].std))
            self.score.append((result['score'].step, result['score'].sum))

        return result

    def finalize(self):
        self.ext_reward = self._finalize_value(self.ext_reward, ['step', 'mean'])
        self.int_reward = self._finalize_value(self.int_reward, ['step', 'max', 'mean', 'std'])
        self.error = self._finalize_value(self.error, ['step', 'max', 'mean', 'std'])
        self.score = self._finalize_value(self.score, ['step', 'sum'])

    @staticmethod
    def _finalize_value(value, keys):
        l = list(zip(*value))
        result = {}
        for i, k in enumerate(keys):
            result[k] = torch.cat(l[i]).numpy()
        return result


if __name__ == '__main__':
    analytic = CNDAnalytic(8)

    analytic.init(ext_reward=(1,), int_reward=(1,), error=(1,), score=(1,))
    for i in range(10):
        analytic.update(ext_reward=torch.rand((8, 1)), int_reward=torch.rand((8, 1)), error=torch.rand((8, 1)),
                        score=torch.rand((8, 1)))
    analytic.reset([1, 3])
    for i in range(4):
        analytic.update(ext_reward=torch.rand((8, 1)), int_reward=torch.rand((8, 1)), error=torch.rand((8, 1)),
                        score=torch.rand((8, 1)))
    analytic.reset([0, 1, 2])
    analytic.finalize()
    pass
