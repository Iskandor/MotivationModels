import torch

from analytic.GenericCollector import GenericCollector


class RNDAnalytic:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print('Creating the object RNDAnalytic')
            cls._instance = super(RNDAnalytic, cls).__new__(cls)
            cls._instance.collector = GenericCollector()
            cls._instance.global_step = 0
            cls._instance.n_env = 0
            cls._instance.ext_reward = []
            cls._instance.int_reward = []
            cls._instance.error = []
            cls._instance.score = []
            cls._instance.ext_value = []
            cls._instance.int_value = []
            cls._instance.loss_prediction = {}
        return cls._instance

    def init(self, n_env, **kwargs):
        self.collector.init(n_env, **kwargs)
        self.n_env = n_env

    def update(self, **kwargs):
        self.collector.update(**kwargs)

        if 'loss_prediction' in kwargs:
            if self.global_step not in self.loss_prediction:
                self.loss_prediction[self.global_step] = []
            self.loss_prediction[self.global_step].append(kwargs['loss_prediction'].cpu())

    def reset(self, indices):
        result = None
        if len(indices) > 0:
            result = self.collector.reset(indices)

            self.ext_reward.append((result['ext_reward'].step, result['ext_reward'].sum))
            self.int_reward.append((result['int_reward'].step, result['int_reward'].max, result['int_reward'].mean, result['int_reward'].std))
            self.error.append((result['error'].step, result['error'].max, result['error'].mean, result['error'].std))
            self.score.append((result['score'].step, result['score'].sum))
            self.ext_value.append((result['ext_value'].step, result['ext_value'].max, result['ext_value'].mean, result['ext_value'].std))
            self.int_value.append((result['int_value'].step, result['int_value'].max, result['int_value'].mean, result['int_value'].std))

        return result

    def end_step(self):
        self.global_step += self.n_env

    def finalize(self):
        self.ext_reward = self._finalize_value(self.ext_reward, ['step', 'sum'], mode='cumsum_step')
        self.int_reward = self._finalize_value(self.int_reward, ['step', 'max', 'mean', 'std'], mode='cumsum_step')
        self.error = self._finalize_value(self.error, ['step', 'max', 'mean', 'std'], mode='cumsum_step')
        self.score = self._finalize_value(self.score, ['step', 'sum'], mode='cumsum_step')
        self.ext_value = self._finalize_value(self.ext_value, ['step', 'max', 'mean', 'std'], mode='cumsum_step')
        self.int_value = self._finalize_value(self.int_value, ['step', 'max', 'mean', 'std'], mode='cumsum_step')
        self.loss_prediction = self._finalize_value(self.loss_prediction, ['step', 'val'], mode='mean_step')

        data = {
            'score': self.score,
            're': self.ext_reward,
            'ri': self.int_reward,
            'error': self.error,
            'loss_prediction': self.loss_prediction,
            'ext_value': self.ext_value,
            'int_value': self.int_value
        }

        return data

    def clear(self):
        self.collector.clear()
        self.global_step = 0
        self.n_env = 0
        self.ext_reward = []
        self.int_reward = []
        self.error = []
        self.score = []
        self.ext_value = []
        self.int_value = []
        self.loss_prediction = {}

    @staticmethod
    def _finalize_value(value, keys, mode):
        result = {}

        if mode == 'cumsum_step':
            l = list(zip(*value))
            for i, k in enumerate(keys):
                result[k] = torch.cat(l[i])

            result['step'] = torch.cumsum(result['step'], dim=0)

            for k in keys:
                result[k] = result[k].numpy()

        if mode == 'mean_step':
            l = []
            for vk in value:
                steps = torch.tensor([[vk]])
                val = torch.stack(value[vk]).mean().unsqueeze(0).unsqueeze(1)
                l.append((steps, val))

            l = list(zip(*l))
            for i, k in enumerate(keys):
                result[k] = torch.cat(l[i])

        return result
