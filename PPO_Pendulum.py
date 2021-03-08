import gym
import torch

from agents import TYPE
from agents.PPOAgent import PPOAgent
from experiment.ppo_experiment import ExperimentPPO


def encode(state):
    return torch.tensor(state, dtype=torch.float32).flatten()


def run_baseline(config, i):
    env = gym.make('Pendulum-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentPPO('Pendulum-v0', env, config)
    experiment.add_preprocess(encode)

    agent = PPOAgent(state_dim, action_dim, config, TYPE.continuous)
    experiment.run_baseline(agent, i)

    env.close()
