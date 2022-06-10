import matplotlib.pyplot as plt
import seaborn as sns
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

    def __init__(self, device='cpu'):
        self.device = device

    def compute(self, samples, epochs):
        input = torch.matmul(samples.unsqueeze(2), samples.unsqueeze(2).permute(0, 2, 1))
        batch = input.shape[0]
        dim = input.shape[1]
        target = torch.ones(batch, device=self.device)

        plt.figure(figsize=(20.48, 20.48))
        sns.heatmap(input[0].cpu().numpy(), cmap='coolwarm')
        plt.show()

        metric_tensor = torch.rand((dim, dim), device=self.device)
        # metric_tensor = torch.zeros((dim, dim), device=self.device)
        metric_tensor.requires_grad = True

        # optimizer = torch.optim.SGD([metric_tensor], lr=1e-6, momentum=0.99)
        optimizer = torch.optim.AdamW([metric_tensor], lr=1e-4)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=1e-6, T_0=2, T_mult=2)

        pbar = tqdm(range(epochs), total=epochs)
        for epoch in pbar:
            optimizer.zero_grad()
            output = (input * metric_tensor).sum(dim=[1, 2])
            loss = nn.functional.mse_loss(output, target, reduction='mean')
            loss.backward()
            optimizer.step()
            # scheduler.step()
            pbar.set_description('{0:d} loss {1:f}'.format(epoch, loss.item()))

        test = (input * metric_tensor).sum(dim=[1, 2])
        for i, t in enumerate(test):
            print(i, t.item())

        return metric_tensor.detach()

    def collect_samples(self, agent, env, count):
        samples = []
        state0 = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0).to(self.device)

        features0, _, action0, probs0 = agent.get_action(state0)
        for _ in tqdm(range(count), total=count):
            action0 = agent.convert_action(action0.cpu()).item()
            next_state, _, _, _ = env.step(action0)
            state0 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
            features1, _, action0, probs0 = agent.get_action(state0)
            samples.append(features0 - features1)
            features0 = features1

        return torch.stack(samples).squeeze(1)


def collect_samples(count):
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
    samples = metric_tensor.collect_samples(agent, env, count)
    np.save('./cnd', samples.cpu().numpy())


def compute_metric_tensor():
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    metric_tensor = MetricTensor('cuda:0')
    samples = torch.tensor(np.load('./cnd.npy'), device='cuda:0')
    result = metric_tensor.compute(samples, epochs=6000)

    plt.figure(figsize=(20.48, 20.48))
    sns.heatmap(result.cpu().numpy(), cmap='coolwarm')
    plt.show()
    # plt.savefig('./metric_tensor.png')


if __name__ == '__main__':
    # collect_samples(1000)
    compute_metric_tensor()
