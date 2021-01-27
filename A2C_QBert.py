import gym
import torch

from experiment.a2c_experiment import ExperimentA2C
from algorithms.A2C import A2C
from utils.AtariWrapper import AtariWrapper


class A2CNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(A2CNetwork, self).__init__()

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
        features = self.feature(state).squeeze(0)
        policy = self.actor(features)
        return policy

    def value(self, state):
        features = self.feature(state).squeeze(0)
        value = self.critic(features)
        return value


def run_baseline(config, i):
    env = AtariWrapper(gym.make('Qbert-v0'))
    state_dim = 4
    action_dim = env.action_space.n

    experiment = ExperimentA2C('Qbert-v0', env, config)

    network = A2CNetwork(state_dim, action_dim, config).to(config.device)
    agent = A2C(network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.beta, config.gamma, config.batch_size, device=config.device)
    experiment.run_baseline(agent, i)

    env.close()

