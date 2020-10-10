import gym
import torch
import numpy as np

from a2c_experiment import ExperimentA2C
from algorithms.A2C import A2C


class A2CNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(A2CNetwork, self).__init__()

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(state_dim, config.critic.h1),
            torch.nn.ReLU(),
            torch.nn.Linear(config.critic.h1, config.critic.h2),
            torch.nn.ReLU(),
            torch.nn.Linear(config.critic.h2, 1)
        )

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(state_dim, config.actor.h1),
            torch.nn.ReLU(),
            torch.nn.Linear(config.actor.h1, config.actor.h2),
            torch.nn.ReLU(),
            torch.nn.Linear(config.actor.h2, action_dim)
        )

    def forward(self, state):
        policy = torch.softmax(self.actor(state), dim=-1)
        value = self.critic(state)
        return policy, value


def run_baseline(config):
    env = gym.make('CartPole-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    experiment = ExperimentA2C('CartPole-v0', env, config)

    for i in range(config.trials):
        network = A2CNetwork(state_dim, action_dim, config)
        agent = A2C(network.actor, network.critic, config.actor.lr, config.critic.lr, config.gamma)
        experiment.run_baseline(agent, i)

    env.close()

