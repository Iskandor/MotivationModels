import torch
import torch.nn as nn
import numpy as np

from agents import TYPE
from modules import init_orthogonal, init_general_wb
from modules.PPO_Modules import Actor, DiscreteHead, Critic2Heads


class Aggregator(nn.Module):
    def __init__(self, n_env, feature_dim, state_dim, frequency, device):
        super(Aggregator, self).__init__()

        self.state_dim = state_dim
        self.frequency = frequency
        self.index = torch.zeros((n_env,), dtype=torch.int32, device=device)

        self.memory = nn.GRUCell(feature_dim, state_dim, device=device)
        self.model = nn.Linear(state_dim, state_dim)
        init_general_wb(nn.init.orthogonal_, self.memory.weight_hh, self.memory.bias_hh, 0.1)
        init_general_wb(nn.init.orthogonal_, self.memory.weight_ih, self.memory.bias_ih, 0.1)

        self.output = torch.zeros((n_env, state_dim), device=device, dtype=torch.float32)
        self.context = torch.zeros((n_env, state_dim), device=device, dtype=torch.float32)
        self.rewards = np.zeros((n_env, 1))
        self.masks = np.zeros((n_env, 1))

    def forward(self, state):
        x = self.memory(state, self.context)
        self.context = x

        mask = (self.index == 0).unsqueeze(1).repeat(1, self.state_dim)
        self.output = self.output * ~mask + self.model(torch.relu(x)) * mask

        return self.output

    def reset(self, indices):
        self.index += 1
        indices += self.indices(self.frequency)

        if len(indices) > 0:
            self.context[indices] *= 0.
            self.rewards[indices] = 0
            self.masks[indices] = 0
            self.index[indices] = 0

    def indices(self, trigger):
        return (self.index == trigger).nonzero().flatten().cpu().tolist()

    def add_reward(self, reward):
        self.rewards += reward
        return self.rewards / (self.index.unsqueeze(1).cpu().numpy() + 1)

    def add_mask(self, mask):
        self.masks = np.maximum(self.masks, mask)
        return self.masks

class DOPControllerAtari(nn.Module):
    def __init__(self, feature_dim, state_dim, action_dim, config):
        super(DOPControllerAtari, self).__init__()

        self.aggregator = Aggregator(config.n_env, feature_dim, state_dim, config.dop_frequency, config.device)

        self.critic = nn.Sequential(
            torch.nn.Linear(state_dim, state_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(state_dim, 1)
        )

        init_orthogonal(self.critic[0], 0.1)
        init_orthogonal(self.critic[2], 0.01)

        self.actor = nn.Sequential(
            torch.nn.Linear(state_dim, state_dim),
            torch.nn.ReLU(),
            DiscreteHead(state_dim, action_dim)
        )

        init_orthogonal(self.actor[0], 0.01)
        init_orthogonal(self.actor[2], 0.01)

        self.actor = Actor(self.actor, TYPE.discrete, action_dim)

    def state(self, features):
        return self.aggregator(features)

    def forward(self, features):
        value = self.critic(features)
        action, probs = self.actor(features)
        action = self.actor.encode_action(action)

        return value, action, probs

    def aggregator_indices(self):
        return self.aggregator.indices(trigger=0)

    def aggregate_values(self, reward, mask):
        return self.aggregator.add_reward(reward), self.aggregator.add_mask(mask)


class DOPActorAtari(nn.Module):
    def __init__(self, head_count, state_dim, action_dim, actor, critic):
        super(DOPActorAtari, self).__init__()

        self.head_count = head_count
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = actor
        self.critic = critic

    def forward(self, features):
        value = self.critic(features).view(-1, self.head_count, 2)
        action, probs = self.actor(features)
        action = self.actor.encode_action(action.view(-1, 1)).view(-1, self.head_count, self.action_dim)

        return value, action, probs

    def select_action(self, index, all_action, all_probs):
        index = index.argmax(dim=1, keepdim=True)
        all_action = all_action.view(-1, self.head_count, self.action_dim)
        all_probs = all_probs.view(-1, self.head_count, self.action_dim)
        action = all_action.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, self.action_dim)).squeeze(1)
        probs = all_probs.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, self.action_dim)).squeeze(1)
        return action, probs


class DOPGeneratorAtari(nn.Module):
    def __init__(self, head_count, state_dim, action_dim, features, actor, critic):
        super(DOPGeneratorAtari, self).__init__()

        self.head_count = head_count
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.features = features
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        features = self.features(state)
        values = self.critic(features)
        action, probs = self.actor(features)
        action = self.actor.encode_action(action.view(-1, 1)).view(-1, self.head_count, self.action_dim)

        return values.unsqueeze(-1), action, probs


class DOPModelAtari(nn.Module):
    def __init__(self, head_count, state_dim, action_dim, config, features, actor, motivator):
        super(DOPModelAtari, self).__init__()

        self.head_count = head_count
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.motivator = motivator
        self.features = features
        self.actor = actor
        self.eta = config.motivation_eta

    def forward(self, state, action):
        predicted_code = self.motivator(state, action)
        return predicted_code

    def error(self, state, action):
        return self.motivator.error(state, action)

    def motivator_loss_function(self, state, action):
        return self.motivator.loss_function(state, action)

    def generator_loss_function(self, state):
        channels = state.shape[1]
        h = state.shape[2]
        w = state.shape[3]

        features = self.features(state)
        action, prob = self.actor(features)

        prob = prob.view(-1, self.action_dim)
        action = action.view(-1, 1)
        state = state.unsqueeze(1).repeat(1, self.head_count, 1, 1, 1).view(-1, channels, h, w)

        logprob = self.actor.log_prob(prob, action)
        action = self.actor.encode_action(action)
        error = self.error(state, action)

        loss = logprob * error * self.eta

        return -loss.mean()
