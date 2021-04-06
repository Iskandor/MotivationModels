import gym
import gym_aeris.envs

from agents.DDPGAgent import DDPGAerisAgent, DDPGAerisForwardModelAgent, DDPGAerisForwardModelEncoderAgent, DDPGAerisInverseModelAgent, DDPGAerisM2ModelAgent, DDPGAerisForwardInverseModelAgent, \
    DDPGAerisGatedMetacriticModelAgent, DDPGAerisRNDModelAgent
from experiment.ddpg_experiment import ExperimentDDPG


def reward_transform(reward):
    r = reward
    if reward < 0:
        r = 0
    return r


def run_baseline(config, i):
    env = gym_aeris.envs.AvoidHazardsEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AvoidHazards-v0', env, config)
    experiment.add_reward_transform(reward_transform)

    agent = DDPGAerisAgent(state_dim, action_dim, config)
    experiment.run_baseline(agent, i)

    env.close()


def run_forward_model(config, i):
    env = gym_aeris.envs.AvoidHazardsEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AvoidHazards-v0', env, config)
    experiment.add_reward_transform(reward_transform)

    agent = DDPGAerisForwardModelAgent(state_dim, action_dim, config)
    experiment.run_forward_model(agent, i)

    env.close()


def run_forward_model_encoder(config, i):
    env = gym_aeris.envs.AvoidHazardsEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AvoidHazards-v0', env, config)
    experiment.add_reward_transform(reward_transform)

    agent = DDPGAerisForwardModelEncoderAgent(state_dim, action_dim, config)
    experiment.run_forward_model_encoder(agent, i)

    env.close()


def run_inverse_model(config, i):
    env = gym_aeris.envs.AvoidHazardsEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AvoidHazards-v0', env, config)
    experiment.add_reward_transform(reward_transform)

    agent = DDPGAerisInverseModelAgent(state_dim, action_dim, config)
    experiment.run_inverse_model(agent, i)

    env.close()


def run_forward_inverse_model(config, i):
    env = gym_aeris.envs.AvoidHazardsEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AvoidHazards-v0', env, config)
    experiment.add_reward_transform(reward_transform)

    agent = DDPGAerisForwardInverseModelAgent(state_dim, action_dim, config)
    experiment.run_forward_inverse_model(agent, i)

    env.close()


def run_m2_model(config, i):
    env = gym_aeris.envs.AvoidHazardsEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AvoidHazards-v0', env, config)
    experiment.add_reward_transform(reward_transform)

    agent = DDPGAerisM2ModelAgent(state_dim, action_dim, config)
    experiment.run_m2_model(agent, i)

    env.close()


def run_rnd_model(config, i):
    env = gym_aeris.envs.AvoidHazardsEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AvoidHazards-v0', env, config)
    experiment.add_reward_transform(reward_transform)

    agent = DDPGAerisRNDModelAgent(state_dim, action_dim, config)
    experiment.run_rnd_model(agent, i)

    env.close()


def run_metalearner_model(config, i):
    env = gym_aeris.envs.AvoidHazardsEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AvoidHazards-v0', env, config)
    experiment.add_reward_transform(reward_transform)

    agent = DDPGAerisGatedMetacriticModelAgent(state_dim, action_dim, config)
    experiment.run_metalearner_model(agent, i)

    env.close()
