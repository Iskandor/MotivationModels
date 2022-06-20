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


class SampleDataset(Dataset):
    def __init__(self, inputs):
        self.buffer = []

        for x in inputs:
            i = torch.matmul(x.unsqueeze(1), x.unsqueeze(1).t())

            # plt.figure(figsize=(20.48, 20.48))
            # sns.heatmap(i.numpy(), cmap='coolwarm')
            # plt.show()

            t = torch.ones(1).squeeze(0)
            self.buffer.append((i, t))

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        i, t = self.buffer[idx]
        return i, t


class SampleCollector:
    def __init__(self, device):
        self.device = device

    def collect_samples_baseline(self, agent, states, next_states):
        samples = []

        network = agent.network.features

        for state, next_state in tqdm(zip(states, next_states), total=len(states)):
            with torch.no_grad():
                features0 = network(torch.tensor([state], device=self.device))
                features1 = network(torch.tensor([next_state], device=self.device))
            samples.append(features0.cpu() - features1.cpu())

        return torch.stack(samples).squeeze(1).numpy()

    def collect_samples_rnd(self, agent, states, next_states):
        samples = []

        network = agent.network.rnd_model

        for state, next_state in tqdm(zip(states, next_states), total=len(states)):
            with torch.no_grad():
                _, features0 = network(torch.tensor([state], device=self.device))
                _, features1 = network(torch.tensor([next_state], device=self.device))
            samples.append(features0.cpu() - features1.cpu())

        return torch.stack(samples).squeeze(1).numpy()

    def collect_samples_cnd(self, agent, states, next_states):
        samples = []

        network = agent.network.cnd_model

        for state, next_state in tqdm(zip(states, next_states), total=len(states)):
            with torch.no_grad():
                _, features0 = network(torch.tensor([state], device=self.device))
                _, features1 = network(torch.tensor([next_state], device=self.device))
            samples.append(features0.cpu() - features1.cpu())

        return torch.stack(samples).squeeze(1).numpy()

    def collect_states(self, agent, env, count):
        states = []
        next_states = []
        state0 = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0).to(self.device)

        _, _, action0, _ = agent.get_action(state0)
        for _ in tqdm(range(count), total=count):
            action0 = agent.convert_action(action0.cpu()).item()
            next_state, _, done, _ = env.step(action0)

            if done:
                next_state = env.reset()

            state1 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
            _, _, action0, _ = agent.get_action(state1)

            if not done:
                states.append(state0.cpu())
                next_states.append(state1.cpu())

            state0 = state1

        return torch.stack(states).squeeze(1).numpy(), torch.stack(next_states).squeeze(1).numpy()


class MetricTensor:

    def __init__(self, device='cpu'):
        self.device = device

    def compute(self, dataloader, dim, epochs):
        metric_tensor = torch.rand((dim, dim), device=self.device)
        # metric_tensor = torch.zeros((dim, dim), device=self.device)
        metric_tensor.requires_grad = True

        # optimizer = torch.optim.SGD([metric_tensor], lr=1e-6, momentum=0.99)
        optimizer = torch.optim.AdamW([metric_tensor], lr=1e-3)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=1e-6, T_0=2, T_mult=2)

        pbar = tqdm(range(epochs), total=epochs)
        for epoch in pbar:
            total_loss = []
            for input, target in dataloader:
                output = (input.to(self.device) * metric_tensor).sum(dim=[1, 2])

                optimizer.zero_grad()
                loss = nn.functional.mse_loss(output, target.to(self.device), reduction='mean')
                loss.backward()
                optimizer.step()
                # scheduler.step()

                total_loss.append(loss.detach())

            total_loss = torch.mean(torch.stack(total_loss)).item()
            pbar.set_description('Epoch {0:d} loss {1:f}'.format(epoch, total_loss))

        for input, target in dataloader:
            test = (input.to(self.device) * metric_tensor).sum(dim=[1, 2])
            for t in test:
                print(t.item())

        return metric_tensor.detach()


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
    states, next_states = collector.collect_states(agent, env, count)
    np.save('./states', {'states': states, 'next_states': next_states})


def collect_samples(agent, path, dest, mode='baseline'):
    data = np.load(path, allow_pickle=True).item()

    states = data['states']
    next_states = data['next_states']

    collector = SampleCollector('cuda:0')

    samples = None
    if mode == 'baseline':
        samples = collector.collect_samples_baseline(agent, states, next_states)
    if mode == 'rnd':
        samples = collector.collect_samples_baseline(agent, states, next_states)
    if mode == 'cnd':
        samples = collector.collect_samples_baseline(agent, states, next_states)
    np.save(dest, samples)


def compute_metric_tensor(path, dest, epochs, batch=1024):
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    metric_tensor = MetricTensor('cuda:0')
    dataset = SampleDataset(torch.tensor(np.load(path)))
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
    result = metric_tensor.compute(dataloader, dim=512, epochs=epochs)

    plt.figure(figsize=(20.48, 20.48))
    sns.heatmap(result.cpu().numpy(), cmap='coolwarm', vmin=-2., vmax=2.)
    # plt.show()
    plt.savefig(dest)


if __name__ == '__main__':
    agent, env, config = initialize_cnd()
    collect_states(agent, env, 10000)

    agent, env, config = initialize_base()
    collect_samples(agent, './states.npy', './base', 'baseline')
    agent, env, config = initialize_rnd()
    collect_samples(agent, './states.npy', './rnd', 'rnd')
    agent, env, config = initialize_cnd()
    collect_samples(agent, './states.npy', './cnd', 'cnd')

    compute_metric_tensor('./base.npy', './tensor_base.png', epochs=50000, batch=10000)
    compute_metric_tensor('./rnd.npy', './tensor_rnd.png', epochs=50000, batch=10000)
    compute_metric_tensor('./cnd.npy', './tensor_cnd.png', epochs=50000, batch=10000)
