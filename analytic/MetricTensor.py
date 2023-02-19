import time

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import gym
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from agents import TYPE
from analytic.StateCollector import SampleDataset
from config import load_config_file
from config.Config import Config
from utils.AtariWrapper import WrapperHardAtari

import os

from utils.ProcgenWrapper import WrapperProcgenExploration

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MetricTensor:

    def __init__(self, device='cpu'):
        self.device = device

    def compute(self, dataloader, dim, lr):
        metric_tensor = torch.rand((dim, dim), device=self.device)
        # metric_tensor = torch.zeros((dim, dim), device=self.device)
        metric_tensor.requires_grad = True

        # optimizer = torch.optim.SGD([metric_tensor], lr=1e-6, momentum=0.99)
        optimizer = torch.optim.AdamW([metric_tensor], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=lr * 1e-1, T_0=2, T_mult=2)

        epoch = 0
        total_loss = 1
        start = time.time()

        while total_loss > 1e-1:
            total_loss = []

            for input, target in dataloader:
                input = input.unsqueeze(1)
                target = torch.pow(target, 2)

                output = torch.bmm(torch.matmul(input, metric_tensor), input.permute(0, 2, 1)).squeeze(-1)

                optimizer.zero_grad()
                loss = nn.functional.mse_loss(output, target, reduction='mean')
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss.append(loss.detach())

            total_loss = torch.mean(torch.stack(total_loss)).item()

            if epoch % 1000 == 0:
                end = time.time()
                print('Epoch {0:d} loss {1:f} time {2:f}s'.format(epoch, total_loss, end - start))
                start = time.time()
            epoch += 1

        print('Epoch {0:d} loss {1:f}'.format(epoch, total_loss))
        # for input, target in dataloader:
        #     input = input.to(self.device).unsqueeze(1)
        #     test = torch.bmm(torch.matmul(input, metric_tensor), input.permute(0, 2, 1)).squeeze(-1)
        #     for t in test:
        #         print(t.item())

        return metric_tensor.detach()

    def load(self, path):
        return np.load(path)

    def volume(self, metric_tensor):
        # volume = np.sqrt(np.abs(np.linalg.det(metric_tensor)))
        sign, logdet = np.linalg.slogdet(metric_tensor)
        volume = sign * np.sqrt(np.abs(logdet))
        return volume


def initialize(env_name, env_id, path, config_id, agent_class):
    config = load_config_file('ppo')

    config = Config(config[env_name][config_id], "{0}_{1}".format(env_name, config_id))
    config.device = 'cuda:0'

    # env = WrapperProcgenExploration(env_id)
    env = WrapperHardAtari(gym.make(env_id))
    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    agent = agent_class(input_shape, action_dim, config, TYPE.discrete)
    agent.load(path)

    return agent, env, config


# def compute_metric_tensor(path, dest, lr, batch=1024, gpu=0):
#     device = 'cuda:{0:d}'.format(gpu)
#     metric_tensor = MetricTensor(device)
#     data = np.load(path, allow_pickle=True).item()
#     dataset = SampleDataset(torch.tensor(data['samples'], device=device, dtype=torch.float32), torch.tensor(data['dist'], device=device, dtype=torch.float32))
#     dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
#     result = metric_tensor.compute(dataloader, dim=512, lr=lr)
#
#     plt.figure(figsize=(20.48, 20.48))
#     sns.heatmap(result.cpu().numpy(), cmap='coolwarm')
#     # plt.show()
#     plt.savefig(dest + '.png')
#     np.save(dest, result.cpu().numpy())

def compute_metric_tensor(src, dst, dest, lr, batch=1024, gpu=0):
    device = 'cuda:{0:d}'.format(gpu)
    metric_tensor = MetricTensor(device)
    src_data = np.load(src, allow_pickle=True).item()
    dst_data = np.load(dst, allow_pickle=True).item()
    dataset = SampleDataset(torch.tensor(dst_data['samples'], device=device, dtype=torch.float32), torch.tensor(src_data['dist'], device=device, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
    result = metric_tensor.compute(dataloader, dim=512, lr=lr)

    plt.figure(figsize=(20.48, 20.48))
    sns.heatmap(result.cpu().numpy(), cmap='coolwarm')
    # plt.show()
    plt.savefig(dest + '.png')
    np.save(dest, result.cpu().numpy())


def plot_tensors(paths):
    data = []

    for path in paths:
        data.append(np.load(path))

    data = np.stack(data)
    # data = (data - np.mean(data)) / np.std(data)
    max_value = np.amax(data)
    min_value = np.amin(data)

    plt.figure(figsize=(len(paths) * 10.24, 10.24))

    for i, path in enumerate(paths):
        ax = plt.subplot(1, len(paths), i+1)
        ax.set_title(path)
        sns.heatmap(data[i], cmap='coolwarm', ax=ax, vmax=max_value, vmin=min_value)
    plt.savefig('metric_tensors.png')

