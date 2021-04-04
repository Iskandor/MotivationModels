import gym
import retro
import torch
import numpy as np

from algorithms.PPO import PPO
from experiment.ppo_nenv_experiment import ExperimentNEnvPPO
from modules import ARCH
from experiment.ppo_experiment import ExperimentPPO
from modules.forward_models.ForwardModel import ForwardModel
from motivation.ForwardModelMotivation import ForwardModelMotivation


def encode_state(state):
    return torch.tensor(np.transpose(state, (2, 0, 1)), dtype=torch.float32).unsqueeze(0)
