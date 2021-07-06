import gym
import pybullet_envs

from agents.DDPGAgent import DDPGBulletAgent, DDPGBulletForwardModelAgent, DDPGBulletGatedMetacriticModelAgent
from algorithms.DDPG import DDPG
from algorithms.DDPG import DDPG
from algorithms.ReplayBuffer import ExperienceReplayBuffer
from experiment.ddpg_experiment import ExperimentDDPG
from modules.DDPG_Modules import *
from modules.forward_models.ForwardModel import ForwardModel
from modules.forward_models.VAE_ForwardModel import VAE_ForwardModel
from modules.metacritic_models.MetacriticRobotic import MetaCriticRobotic
from motivation.ForwardModelMotivation import ForwardModelMotivation
from motivation.MetaCriticMotivation import MetaCriticMotivation
from motivation.VAE_ForwardModelMotivation import VAE_ForwardModelMotivation


def run_baseline(config, i):
    env = gym.make('AntBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AntBulletEnv-v0', env, config)

    agent = DDPGBulletAgent(state_dim, action_dim, config)
    experiment.run_baseline(agent, i)

    env.close()


def run_forward_model(config, i):
    env = gym.make('AntBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AntBulletEnv-v0', env, config)
    agent = DDPGBulletForwardModelAgent(state_dim, action_dim, config)

    experiment.run_forward_model(agent, i)

    env.close()


def run_vae_forward_model(config, i):
    pass


def run_metalearner_model(config, i):
    env = gym.make('AntBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AntBulletEnv-v0', env, config)
    agent = DDPGBulletGatedMetacriticModelAgent(state_dim, action_dim, config)

    experiment.run_metalearner_model(agent, i)

    env.close()
