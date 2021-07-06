import gym
import pybullet_envs
from algorithms.DDPG import DDPG
from algorithms.ReplayBuffer import ExperienceReplayBuffer
from experiment.ddpg_experiment import ExperimentDDPG
from modules.DDPG_Modules import *
from modules.forward_models.ForwardModel import ForwardModel
from modules.metacritic_models import MetaCritic
from motivation.ForwardModelMotivation import ForwardModelMotivation
from motivation.MetaCriticMotivation import MetaCriticMotivation


def run_baseline(config, i):
    pass


def run_forward_model(config, i):
    pass


def run_metalearner_model(config, i):
    pass
