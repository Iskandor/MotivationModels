import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def collect_states(agent, env, count):
    collector = SampleCollector('cuda:0')
    states, next_states = collector.collect_states(agent, env, count)
    return states, next_states


def save_states(path, states, next_states):
    states = np.concatenate(states)
    next_states = np.concatenate(next_states)

    # batch = states.shape[0]
    # dist = torch.cdist(torch.tensor(states).view(batch, -1), torch.tensor(states).view(batch, -1), p=2)
    # plt.figure(figsize=(20.48, 20.48))
    # sns.heatmap(dist.numpy(), cmap='coolwarm')
    # plt.savefig('./states.png')

    np.save(path, {'states': states, 'next_states': next_states})


def collect_samples(agent, path, dest, mode):
    data = np.load(path, allow_pickle=True).item()

    states = data['states']
    next_states = data['next_states']

    collector = SampleCollector('cuda:0')

    feature = None
    diff = None
    dist = None
    if mode == 'rnd':
        feature, diff, dist = collector.collect_samples(agent.network.rnd_model, states, next_states, False)
    if mode == 'cnd':
        feature, diff, dist = collector.collect_samples(agent.network.cnd_model, states, next_states, False)
    if mode == 'icm':
        feature, diff, dist = collector.collect_samples(agent.network.features, states, next_states, True)
    if mode == 'fwd':
        feature, diff, dist = collector.collect_samples(agent.network.features, states, next_states, True)
    if mode == 'fed_ref':
        feature, diff, dist = collector.collect_samples(agent.network.features, states, next_states, True)

    # d = torch.cdist(torch.tensor(feature), torch.tensor(feature), p=2)
    # plt.figure(figsize=(20.48, 20.48))
    # sns.heatmap(d.numpy(), cmap='coolwarm')
    # plt.savefig('./features.png')

    np.save(dest, {'feature': feature, 'dist': dist})


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

    def collect_samples(self, network, states, next_states, single_output=True):
        features = []
        diff = []

        for state, next_state in tqdm(zip(states, next_states), total=len(states)):
            with torch.no_grad():
                if single_output:
                    features0 = network(torch.tensor(state, device=self.device).unsqueeze(0))
                    features1 = network(torch.tensor(next_state, device=self.device).unsqueeze(0))
                else:
                    _, features0 = network(torch.tensor(state, device=self.device).unsqueeze(0))
                    _, features1 = network(torch.tensor(next_state, device=self.device).unsqueeze(0))

            features.append(features0.cpu())
            diff.append(features1.cpu() - features0.cpu())

        dist = torch.norm(torch.stack(diff).squeeze(1), p=2, dim=1, keepdim=True)

        return torch.stack(features).squeeze(1).numpy(), torch.stack(diff).squeeze(1).numpy(), dist.numpy()

    def collect_states(self, agent, env, count):
        states = []
        next_states = []
        state0 = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0).to(self.device)

        bar = tqdm(total=count + 3)

        result = agent.get_action(state0)
        action0 = result[len(result) - 2]

        while len(states) < count + 3:
            action0 = agent.convert_action(action0.cpu()).item()
            next_state, _, done, _ = env.step(action0)

            if done:
                next_state = env.reset()

            state1 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
            result = agent.get_action(state1)
            action0 = result[len(result) - 2]

            if not done:
                bar.update()
                bar.display()
                states.append(state0.cpu())
                next_states.append(state1.cpu())

            state0 = state1

        states = torch.stack(states[3:]).squeeze(1)
        next_states = torch.stack(next_states[3:]).squeeze(1)

        return states.numpy(), next_states.numpy()
