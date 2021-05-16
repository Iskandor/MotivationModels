import gym
import pybullet_envs

from agents.DDPGAgent import DDPGBulletAgent, DDPGBulletForwardModelAgent, DDPGBulletGatedMetacriticModelAgent, DDPGBulletRNDModelAgent, DDPGBulletMetaCriticRNDModelAgent
from experiment.ddpg_experiment import ExperimentDDPG


def run_baseline(config, i):
    env = gym.make('HopperBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('HopperBulletEnv-v0', env, config)

    agent = DDPGBulletAgent(state_dim, action_dim, config)
    experiment.run_baseline(agent, i)

    env.close()


def run_forward_model(config, i):
    env = gym.make('HopperBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('HopperBulletEnv-v0', env, config)
    agent = DDPGBulletForwardModelAgent(state_dim, action_dim, config)

    experiment.run_forward_model(agent, i)

    env.close()


def run_metalearner_model(config, i):
    env = gym.make('HopperBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('HopperBulletEnv-v0', env, config)
    agent = DDPGBulletGatedMetacriticModelAgent(state_dim, action_dim, config)

    experiment.run_metalearner_model(agent, i)

    env.close()


def run_rnd_model(config, i):
    env = gym.make('HopperBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('HopperBulletEnv-v0', env, config)
    agent = DDPGBulletRNDModelAgent(state_dim, action_dim, config)
    experiment.run_rnd_model(agent, i)

    env.close()


def run_metalearner_rnd_model(config, i):
    env = gym.make('HopperBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('HopperBulletEnv-v0', env, config)
    agent = DDPGBulletMetaCriticRNDModelAgent(state_dim, action_dim, config)
    experiment.run_metalearner_rnd_model(agent, i)

    env.close()
