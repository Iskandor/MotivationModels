import time

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import gym
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from agents import TYPE
from agents.PPOAtariAgent import PPOAtariCNDAgent, PPOAtariRNDAgent
from config import load_config_file
from config.Config import Config
from utils.AtariWrapper import WrapperHardAtari

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class SampleDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        i = self.inputs[idx]
        t = self.targets[idx]
        return i, t


class SampleCollector:
    def __init__(self, device):
        self.device = device

    def collect_samples_baseline(self, agent, states, next_states):
        samples = []

        network = agent.network.features

        for state, next_state in tqdm(zip(states, next_states), total=len(states)):
            with torch.no_grad():
                features0 = network(torch.tensor(state, device=self.device).unsqueeze(0))
                features1 = network(torch.tensor(next_state, device=self.device).unsqueeze(0))
            samples.append(features1.cpu() - features0.cpu())

        return torch.stack(samples).squeeze(1).numpy()

    def collect_samples_rnd(self, agent, states, next_states):
        samples = []

        network = agent.network.rnd_model

        for state, next_state in tqdm(zip(states, next_states), total=len(states)):
            with torch.no_grad():
                _, features0 = network(torch.tensor(state, device=self.device).unsqueeze(0))
                _, features1 = network(torch.tensor(next_state, device=self.device).unsqueeze(0))
            samples.append(features1.cpu() - features0.cpu())

        return torch.stack(samples).squeeze(1).numpy()

    def collect_samples_cnd(self, agent, states, next_states):
        samples = []

        network = agent.network.cnd_model

        for state, next_state in tqdm(zip(states, next_states), total=len(states)):
            with torch.no_grad():
                _, features0 = network(torch.tensor(state, device=self.device).unsqueeze(0))
                _, features1 = network(torch.tensor(next_state, device=self.device).unsqueeze(0))
            samples.append(features1.cpu() - features0.cpu())

        return torch.stack(samples).squeeze(1).numpy()

    def collect_states(self, agent, env, count):
        states = []
        next_states = []
        state0 = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0).to(self.device)

        bar = tqdm(total=count + 3)

        _, _, action0, _ = agent.get_action(state0)
        while len(states) < count + 3:
            action0 = agent.convert_action(action0.cpu()).item()
            next_state, _, done, _ = env.step(action0)

            if done:
                next_state = env.reset()

            state1 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
            _, _, action0, _ = agent.get_action(state1)

            if not done:
                bar.update()
                bar.display()
                states.append(state0.cpu())
                next_states.append(state1.cpu())

            state0 = state1

        states = torch.stack(states[3:]).squeeze(1)
        next_states = torch.stack(next_states[3:]).squeeze(1)
        dist = torch.norm(next_states - states, p=1, dim=[1, 2, 3]).unsqueeze(-1)

        return states.numpy(), next_states.numpy(), dist.numpy()


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
                loss = nn.functional.mse_loss(output, target, reduction='sum')
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


def initialize_cnd():
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

    return agent, env, config


def initialize_rnd():
    env_id = 'montezuma'
    env_name = 'MontezumaRevengeNoFrameskip-v4'
    path = './models/MontezumaRevengeNoFrameskip-v4_rnd_0'
    config_id = '2'
    config = load_config_file('ppo')

    config = Config(config[env_id][config_id], "{0}_{1}".format(env_id, config_id))
    config.device = 'cuda:0'

    env = WrapperHardAtari(gym.make(env_name))
    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    agent = PPOAtariRNDAgent(input_shape, action_dim, config, TYPE.discrete)
    agent.load(path)

    return agent, env, config


def initialize_base():
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

    return agent, env, config


def collect_states(agent, env, count):
    collector = SampleCollector('cuda:0')
    states, next_states, dist = collector.collect_states(agent, env, count)
    np.save('./states', {'states': states, 'next_states': next_states, 'dist': dist})


def collect_samples(agent, path, dest, mode='baseline'):
    data = np.load(path, allow_pickle=True).item()

    states = data['states']
    next_states = data['next_states']
    dist = data['dist']

    collector = SampleCollector('cuda:0')

    samples = None
    if mode == 'baseline':
        samples = collector.collect_samples_baseline(agent, states, next_states)
    if mode == 'rnd':
        samples = collector.collect_samples_rnd(agent, states, next_states)
    if mode == 'cnd':
        samples = collector.collect_samples_cnd(agent, states, next_states)
    np.save(dest, {'samples': samples, 'dist': dist})


def compute_metric_tensor(path, dest, lr, batch=1024, gpu=0):
    device = 'cuda:{0:d}'.format(gpu)
    metric_tensor = MetricTensor(device)
    data = np.load(path, allow_pickle=True).item()
    dataset = SampleDataset(torch.tensor(data['samples'], device=device, dtype=torch.float32), torch.tensor(data['dist'], device=device, dtype=torch.float32))
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

