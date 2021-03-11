import gym
import torch

from agents import TYPE
from agents.PPOAgent import PPOAtariAgent
from experiment.ppo_experiment import ExperimentPPO
from utils.AtariWrapper import WrapperAtari


def encode_state(state):
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0)


def run_baseline(config, i):
    env = WrapperAtari(gym.make('Qbert-v0'))
    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    experiment = ExperimentPPO('Qbert-v0', env, config)
    experiment.add_preprocess(encode_state)

    agent = PPOAtariAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_baseline(agent, i)

    env.close()
