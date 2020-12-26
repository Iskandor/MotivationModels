import gym
import torch

from algorithms.PPO import PPO
from motivation.ICM import ICM
from ppo_experiment import ExperimentPPO
from utils.AtariWrapper import AtariWrapper
from nes_py.wrappers import JoypadSpace
import gym_zelda_1
from gym_zelda_1.actions import MOVEMENT


class PPONetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(PPONetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = 64 * 6 * 6

        self.feature = torch.nn.Sequential(
            torch.nn.Conv2d(state_dim, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten()
        )
        self.feature.apply(self.init_weights)

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, config.critic_h1),
            torch.nn.ReLU(),
            torch.nn.Linear(config.critic_h1, config.critic_h2),
            torch.nn.ReLU(),
            torch.nn.Linear(config.critic_h2, 1)
        )
        self.critic.apply(self.init_weights)

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, config.actor_h1),
            torch.nn.ReLU(),
            torch.nn.Linear(config.actor_h1, config.actor_h2),
            torch.nn.ReLU(),
            torch.nn.Linear(config.actor_h2, action_dim)
        )
        self.actor.apply(self.init_weights)

    def init_weights(self, module):
        if type(module) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(module.weight)
        if type(module) == torch.nn.Conv2d:
            torch.nn.init.xavier_uniform_(module.weight)

    def action(self, state):
        features = self.feature(state)
        policy = self.actor(features)
        return policy

    def value(self, state):
        features = self.feature(state)
        value = self.critic(features)
        return value


class ICMNetwork(torch.nn.Module):
    def __init__(self, feature_network, feature_dim, action_dim, config):
        super(ICMNetwork, self).__init__()

        self.feature_network = feature_network

        self._rate = feature_dim // action_dim

        self.forward_model = torch.nn.Sequential(
            torch.nn.Linear(feature_dim + action_dim * self._rate, config.forward_model_h1),
            torch.nn.ReLU(),
            torch.nn.Linear(config.forward_model_h1, config.forward_model_h2),
            torch.nn.ReLU(),
            torch.nn.Linear(config.forward_model_h2, feature_dim)
        )

        self.inverse_model = torch.nn.Sequential(
            torch.nn.Linear(feature_dim + feature_dim, config.inverse_model.h1),
            torch.nn.ReLU(),
            torch.nn.Linear(config.inverse_model.h1, config.inverse_model.h2),
            torch.nn.ReLU(),
            torch.nn.Linear(config.inverse_model.h2, action_dim)
        )

    def get_features(self, state):
        features = self.feature_network(state)
        return features

    def estimate_state(self, state, action):
        features = self.feature_network(state)
        x = torch.cat([features, action.repeat(1,self._rate)], dim=1)
        next_state = self.forward_model(x)
        return next_state

    def estimate_action(self, state, next_state):
        features = self.feature_network(state)
        next_features = self.feature_network(next_state)
        x = torch.cat([features, next_features], dim=1)
        action = self.inverse_model(x)
        return action


def run_baseline(config):
    env = JoypadSpace(gym.make('Zelda1-v0'), MOVEMENT)
    state_dim = 4
    action_dim = env.action_space.n

    experiment = ExperimentPPO('Zelda1-v0', env, config)

    for i in range(config.trials):
        network = PPONetwork(state_dim, action_dim, config).to(config.device)
        agent = PPO(network, config.lr, config.actor.loss_weight, config.critic.loss_weight, config.batch_size, config.trajectory_size, config.beta, config.gamma, device=config.device)
        experiment.run_baseline(agent, i)

    env.close()


def run_icm(config):
    env = gym.make('Zelda1-v0')
    state_dim = 4
    action_dim = env.action_space.n
    config.state_dim = state_dim
    config.action_dim = action_dim

    experiment = ExperimentPPO('Zelda1-v0', env, config)

    for i in range(config.trials):
        network = PPONetwork(state_dim, action_dim, config).to(config.device)
        icm_network = ICMNetwork(network.feature, network.feature_dim, action_dim, config).to(config.device)
        network.add_module('icm', icm_network)
        icm = ICM(icm_network, config.forward_model.beta, config.forward_model_eta)
        agent = PPO(network, config.lr, config.actor.loss_weight, config.critic.loss_weight, config.batch_size, config.trajectory_size, config.beta, config.gamma, device=config.device)
        agent.add_motivation(icm)
        experiment.run_icm(agent, i)

    env.close()

