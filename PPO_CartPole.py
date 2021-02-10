import gym
import torch

from algorithms.PPO import PPO
from experiment.ppo_experiment import ExperimentPPO


class PPONetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(PPONetwork, self).__init__()

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
        if state.ndim == 1:
            state = state.unsqueeze(0)
        policy = self.actor(state)
        return policy

    def value(self, state):
        if state.ndim == 1:
            state = state.unsqueeze(0)
        value = self.critic(state)
        return value


def run_baseline(config, i):
    env = gym.make('CartPole-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    experiment = ExperimentPPO('CartPole-v0', env, config)

    network = PPONetwork(state_dim, action_dim, config)
    agent = PPO(network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size, config.beta, config.gamma)
    experiment.run_baseline(agent, i)

    env.close()
