import torch

from analytic.GenericCollector import GenericCollector


class CNDAnalytic:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print('Creating the object CNDAnalytic')
            cls._instance = super(CNDAnalytic, cls).__new__(cls)
            cls._instance.collector = GenericCollector()
            cls._instance.global_step = 0
            cls._instance.n_env = 0
            cls._instance.ext_reward = []
            cls._instance.int_reward = []
            cls._instance.error = []
            cls._instance.score = []
            cls._instance.state_space = []
            cls._instance.feature_space = []
            cls._instance.ext_value = []
            cls._instance.int_value = []
            cls._instance.loss_prediction = {}
            cls._instance.loss_target = {}
            cls._instance.loss_reg = {}
            cls._instance.loss_target_norm = {}
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
        if 'loss_target' in kwargs:
            if self.global_step not in self.loss_target:
                self.loss_target[self.global_step] = []
            self.loss_target[self.global_step].append(kwargs['loss_target'].cpu())
        if 'loss_reg' in kwargs:
            if self.global_step not in self.loss_reg:
                self.loss_reg[self.global_step] = []
            self.loss_reg[self.global_step].append(kwargs['loss_reg'].cpu())
        if 'loss_target_norm' in kwargs:
            if self.global_step not in self.loss_target_norm:
                self.loss_target_norm[self.global_step] = []
            self.loss_target_norm[self.global_step].append(kwargs['loss_target_norm'].cpu())

    def reset(self, indices):
        result = None
        if len(indices) > 0:
            result = self.collector.reset(indices)

            self.ext_reward.append((result['ext_reward'].step, result['ext_reward'].sum))
            self.int_reward.append((result['int_reward'].step, result['int_reward'].max, result['int_reward'].mean, result['int_reward'].std))
            self.error.append((result['error'].step, result['error'].max, result['error'].mean, result['error'].std))
            self.score.append((result['score'].step, result['score'].sum))
            self.state_space.append((result['state_space'].step, result['state_space'].max, result['state_space'].mean, result['state_space'].std))
            self.feature_space.append((result['feature_space'].step, result['feature_space'].max, result['feature_space'].mean, result['feature_space'].std))
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
        self.state_space = self._finalize_value(self.state_space, ['step', 'max', 'mean', 'std'], mode='cumsum_step')
        self.feature_space = self._finalize_value(self.feature_space, ['step', 'max', 'mean', 'std'], mode='cumsum_step')
        self.ext_value = self._finalize_value(self.ext_value, ['step', 'max', 'mean', 'std'], mode='cumsum_step')
        self.int_value = self._finalize_value(self.int_value, ['step', 'max', 'mean', 'std'], mode='cumsum_step')
        self.loss_prediction = self._finalize_value(self.loss_prediction, ['step', 'val'], mode='mean_step')
        self.loss_target = self._finalize_value(self.loss_target, ['step', 'val'], mode='mean_step')
        self.loss_reg = self._finalize_value(self.loss_reg, ['step', 'val'], mode='mean_step')
        self.loss_target_norm = self._finalize_value(self.loss_target_norm, ['step', 'val'], mode='mean_step')

        data = {
            'score': self.score,
            're': self.ext_reward,
            'ri': self.int_reward,
            'error': self.error,
            'loss_prediction': self.loss_prediction,
            'loss_target': self.loss_target,
            'loss_reg': self.loss_reg,
            'loss_target_norm': self.loss_target_norm,
            'state_space': self.state_space,
            'feature_space': self.feature_space,
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
        self.state_space = []
        self.feature_space = []
        self.ext_value = []
        self.int_value = []
        self.loss_prediction = {}
        self.loss_target = {}
        self.loss_reg = {}
        self.loss_target_norm = {}

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
