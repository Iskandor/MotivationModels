import torch

from analytic.GenericCollector import GenericCollector


class CNDAnalytic:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print('Creating the object CNDAnalytic')
            cls._instance = super(CNDAnalytic, cls).__new__(cls)
            cls._instance.collector = GenericCollector()
            cls._instance.steps = 0
            cls._instance.ext_reward = []
            cls._instance.int_reward = []
            cls._instance.error = []
            cls._instance.score = []
            cls._instance.loss_prediction = []
            cls._instance.loss_target = []
        return cls._instance

    def init(self, n_env, **kwargs):
        self.collector.init(n_env, **kwargs)

    def update(self, **kwargs):
        self.collector.update(**kwargs)

        if 'loss_prediction' in kwargs:
            steps = torch.tensor(self.steps, dtype=torch.int32).unsqueeze(-1)
            self.loss_prediction.append((steps, kwargs['loss_prediction'].cpu()))
        if 'loss_target' in kwargs:
            steps = torch.tensor(self.steps, dtype=torch.int32).unsqueeze(-1)
            self.loss_target.append((steps, kwargs['loss_target'].cpu()))

    def reset(self, indices):
        result = None
        if len(indices) > 0:
            result = self.collector.reset(indices)
            steps = torch.tensor(self.steps, dtype=torch.int32).repeat(len(indices)).unsqueeze(-1)

            self.ext_reward.append((steps, result['ext_reward'].sum))
            self.int_reward.append(
                (steps, result['int_reward'].max, result['int_reward'].mean, result['int_reward'].std))
            self.error.append((steps, result['error'].max, result['error'].mean, result['error'].std))
            self.score.append((steps, result['score'].sum))

        return result

    def end_step(self):
        self.steps += 1

    def finalize(self):
        self.ext_reward = self._finalize_value(self.ext_reward, ['step', 'sum'])
        self.int_reward = self._finalize_value(self.int_reward, ['step', 'max', 'mean', 'std'])
        self.error = self._finalize_value(self.error, ['step', 'max', 'mean', 'std'])
        self.score = self._finalize_value(self.score, ['step', 'sum'])
        self.loss_prediction = self._finalize_value(self.loss_prediction, ['step', 'val'])
        self.loss_target = self._finalize_value(self.loss_target, ['step', 'val'])

    def clear(self):
        self.collector.clear()
        self.steps = 0
        self.ext_reward = []
        self.int_reward = []
        self.error = []
        self.score = []
        self.loss_prediction = []
        self.loss_target = []

    @staticmethod
    def _finalize_value(value, keys):
        l = list(zip(*value))
        result = {}
        for i, k in enumerate(keys):
            result[k] = torch.cat(l[i]).numpy()
        return result


if __name__ == '__main__':
    analytic = CNDAnalytic()

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
