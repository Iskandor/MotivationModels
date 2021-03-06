import gym
import torch

from experiment.a2c_experiment import ExperimentA2C
from algorithms.A2C import A2C


class A2CNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(A2CNetwork, self).__init__()

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(state_dim, config.critic_h1),
            torch.nn.ReLU(),
            torch.nn.Linear(config.critic_h1, config.critic_h2),
            torch.nn.ReLU(),
            torch.nn.Linear(config.critic_h2, 1)
        )
        self.critic.apply(self.init_weights)

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(state_dim, config.actor_h1),
            torch.nn.ReLU(),
            torch.nn.Linear(config.actor_h1, config.actor_h2),
            torch.nn.ReLU(),
            torch.nn.Linear(config.actor_h2, action_dim)
        )
        self.actor.apply(self.init_weights)

    def init_weights(self, module):
        if type(module) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(module.weight)

    def action(self, state):
        policy = self.actor(state)
        return policy

    def value(self, state):
        value = self.critic(state)
        return value


def run_baseline(config, i):
    env = gym.make('CartPole-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    experiment = ExperimentA2C('CartPole-v0', env, config)

    network = A2CNetwork(state_dim, action_dim, config)
    agent = A2C(network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.beta, config.gamma, config.batch_size)
    experiment.run_baseline(agent, i)

    env.close()

