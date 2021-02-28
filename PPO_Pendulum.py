import gym
import torch

from algorithms.PPO import PPO
from experiment.ppo_experiment import ExperimentPPO
from modules.PPO_Modules import PPONetwork, HEAD


def encode(state):
    return torch.tensor(state, dtype=torch.float32).flatten()


def run_baseline(config, i):
    env = gym.make('Pendulum-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentPPO('Pendulum-v0', env, config)
    experiment.add_preprocess(encode)

    network = PPONetwork(state_dim, action_dim, config, head=HEAD.continuous)
    agent = PPO(network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size, config.beta, config.gamma)
    experiment.run_baseline(agent, i)

    env.close()
