import gym
import torch

from algorithms.PPO import PPO
from experiment.ppo_experiment import ExperimentPPO
from modules.PPO_Modules import PPONetwork


def run_baseline(config, i):
    env = gym.make('CartPole-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    experiment = ExperimentPPO('CartPole-v0', env, config)

    network = PPONetwork(state_dim, action_dim, config)
    agent = PPO(network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size, config.beta, config.gamma)
    experiment.run_baseline(agent, i)

    env.close()
