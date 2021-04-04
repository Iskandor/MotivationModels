import gym
import gym_aeris.envs

from agents.DDPGAgent import DDPGAerisAgent, DDPGAerisForwardModelAgent, DDPGAerisForwardModelEncoderAgent, DDPGAerisInverseModelAgent, DDPGAerisM2ModelAgent, DDPGAerisForwardInverseModelAgent, \
    DDPGAerisGatedMetacriticModelAgent, DDPGAerisRNDModelAgent
from experiment.ddpg_experiment import ExperimentDDPG


def run_baseline(config, i):
    env = gym_aeris.envs.AvoidFragilesEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AvoidFragiles-v0', env, config)

    agent = DDPGAerisAgent(state_dim, action_dim, config)
    experiment.run_baseline(agent, i)

    env.close()


def run_forward_model(config, i):
    env = gym_aeris.envs.AvoidFragilesEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AvoidFragiles-v0', env, config)

    agent = DDPGAerisForwardModelAgent(state_dim, action_dim, config)
    experiment.run_forward_model(agent, i)

    env.close()


def run_forward_model_encoder(config, i):
    env = gym_aeris.envs.AvoidFragilesEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AvoidFragiles-v0', env, config)

    agent = DDPGAerisForwardModelEncoderAgent(state_dim, action_dim, config)
    experiment.run_forward_model_encoder(agent, i)

    env.close()


def run_inverse_model(config, i):
    env = gym_aeris.envs.AvoidFragilesEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AvoidFragiles-v0', env, config)

    agent = DDPGAerisInverseModelAgent(state_dim, action_dim, config)
    experiment.run_inverse_model(agent, i)

    env.close()


def run_forward_inverse_model(config, i):
    env = gym_aeris.envs.AvoidFragilesEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AvoidFragiles-v0', env, config)

    agent = DDPGAerisForwardInverseModelAgent(state_dim, action_dim, config)
    experiment.run_forward_inverse_model(agent, i)

    env.close()


def run_m2_model(config, i):
    env = gym_aeris.envs.AvoidFragilesEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AvoidFragiles-v0', env, config)

    agent = DDPGAerisM2ModelAgent(state_dim, action_dim, config)
    experiment.run_m2_model(agent, i)

    env.close()


def run_rnd_model(config, i):
    env = gym_aeris.envs.AvoidFragilesEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AvoidFragiles-v0', env, config)

    agent = DDPGAerisRNDModelAgent(state_dim, action_dim, config)
    experiment.run_rnd_model(agent, i)

    env.close()


def run_metalearner_model(config, i):
    env = gym_aeris.envs.AvoidFragilesEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AvoidFragiles-v0', env, config)

    agent = DDPGAerisGatedMetacriticModelAgent(state_dim, action_dim, config)
    experiment.run_metalearner_model(agent, i)

    env.close()
