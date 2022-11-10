import gym
import pybulletgym

from agents.DDPGAgent import DDPGBulletAgent, DDPGBulletForwardModelAgent, DDPGBulletRNDModelAgent, DDPGBulletQRNDModelAgent, DDPGBulletGatedMetacriticModelAgent, DDPGBulletMetaCriticRNDModelAgent
from experiment.ddpg_experiment import ExperimentDDPG



def run_baseline(env_name, config, trial):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG(env_name, env, config)

    agent = DDPGBulletAgent(state_dim, action_dim, config)
    experiment.run_baseline(agent, trial)

    env.close()


def run_forward_model(env_name, config, trial):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG(env_name, env, config)

    agent = DDPGBulletForwardModelAgent(state_dim, action_dim, config)
    experiment.run_forward_model(agent, trial)

    env.close()


def run_rnd_model(env_name, config, i):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG(env_name, env, config)

    agent = DDPGBulletRNDModelAgent(state_dim, action_dim, config)
    experiment.run_rnd_model(agent, i)

    env.close()


def run_qrnd_model(env_name, config, i):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG(env_name, env, config)

    agent = DDPGBulletQRNDModelAgent(state_dim, action_dim, config)
    experiment.run_qrnd_model(agent, i)

    env.close()


def run_metalearner_model(env_name, config, trial):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG(env_name, env, config)

    agent = DDPGBulletGatedMetacriticModelAgent(state_dim, action_dim, config)
    experiment.run_metalearner_model(agent, trial)

    env.close()


def run_metalearner_rnd_model(env_name, config, trial):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG(env_name, env, config)

    agent = DDPGBulletMetaCriticRNDModelAgent(state_dim, action_dim, config)
    experiment.run_metalearner_rnd_model(agent, trial)

    env.close()
