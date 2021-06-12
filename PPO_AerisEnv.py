import gym
import gym_aeris.envs

from agents import TYPE
from agents.PPOAgent import PPOAerisAgent, PPOAerisRNDAgent
from experiment.ppo_experiment import ExperimentPPO


def run_baseline(env_name, env, config, i):
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentPPO(env_name, env, config)

    agent = PPOAerisAgent(state_dim, action_dim, config, TYPE.continuous)
    experiment.run_baseline(agent, i)

    env.close()

def run_rnd_model(env_name, env, config, i):
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentPPO(env_name, env, config)

    agent = PPOAerisRNDAgent(state_dim, action_dim, config, TYPE.continuous)
    experiment.run_rnd_model(agent, i)

    env.close()
