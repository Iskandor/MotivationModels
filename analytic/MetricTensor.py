import numpy as np

import gym
import torch
import torch.nn as nn
from tqdm import tqdm

from agents import TYPE
from agents.PPOAtariAgent import PPOAtariCNDAgent
from config import load_config_file
from config.Config import Config
from utils.AtariWrapper import WrapperHardAtari


class MetricTensor:

    def __init__(self, device):
        self.device = device
        self.samples = []

    def compute(self):
        input = torch.stack(self.samples).squeeze(1)
        batch = input.shape[0]
        dim = input.shape[1]
        target = torch.ones(batch, device=self.device)

        metric_tensor = torch.rand((dim, dim), device=self.device)
        # metric_tensor = torch.zeros((dim, dim), device=self.device)
        metric_tensor.requires_grad = True

        optimizer = torch.optim.SGD([metric_tensor], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-5)

        epochs = 10000
        pbar = tqdm(range(epochs), total=epochs)
        for epoch in pbar:
            optimizer.zero_grad()
            output = torch.matmul(input, metric_tensor).sum(dim=1)
            loss = nn.functional.mse_loss(output, target, reduction='mean')
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_description('{0:d} loss {1:f}'.format(epoch, loss.item()))

        return metric_tensor

    def collect_samples(self, agent, env, count):
        state0 = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0).to(self.device)

        for _ in tqdm(range(count), total=count):
            features0, _, action0, probs0 = agent.get_action(state0)
            action0 = agent.convert_action(action0.cpu()).item()
            next_state, _, _, _ = env.step(action0)
            state0 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
            features1, _, _, _ = agent.get_action(state0)
            self.samples.append(features0 - features1)


if __name__ == '__main__':
    env_id = 'montezuma'
    env_name = 'MontezumaRevengeNoFrameskip-v4'
    path = './models/montezuma_21_cnd_0'
    config_id = '21'
    config = load_config_file('ppo')

    config = Config(config[env_id][config_id], "{0}_{1}".format(env_id, config_id))
    config.device = 'cuda:0'

    env = WrapperHardAtari(gym.make(env_name))
    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    agent = PPOAtariCNDAgent(input_shape, action_dim, config, TYPE.discrete)
    agent.load(path)

    metric_tensor = MetricTensor(config.device)
    metric_tensor.collect_samples(agent, env, 4500)
    result = metric_tensor.compute()

    print(result)
