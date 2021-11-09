import copy

import gym
import gym_aeris.envs
from gym.spaces import Box, Discrete

from agents import TYPE
from agents.PPOAerisAgent import PPOAerisAgent, PPOAerisRNDAgent, PPOAerisDOPAgent, PPOAerisDOPRefAgent
from experiment.ppo_experiment import ExperimentPPO
from experiment.ppo_nenv_experiment import ExperimentNEnvPPO
from utils.MultiEnvWrapper import MultiEnvParallel


def create_env(env_id):
    env = None
    if env_id == 'AvoidFragiles-v0':
        env = gym_aeris.envs.AvoidFragilesEnv()
    if env_id == 'AvoidHazards-v0':
        env = gym_aeris.envs.AvoidHazardsEnv()
    if env_id == 'TargetNavigate-v0':
        env = gym_aeris.envs.TargetNavigateEnv()
    if env_id == 'GridTargetSearchAEnv-v0':
        env = gym_aeris.envs.GridTargetSearchADiscreteEnv()
    if env_id == 'GridTargetSearchBEnv-v0':
        env = gym_aeris.envs.GridTargetSearchBDiscreteEnv()

    return env


def run_baseline(env_name, config, trial, agent_class):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([create_env(env_name) for _ in range(config.n_env)], config.n_env, config.num_threads)

    input_shape = env.observation_space.shape
    if isinstance(env.action_space, Box):
        action_dim = env.action_space.shape[0]
    if isinstance(env.action_space, Discrete):
        action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO(env_name, env, config)

    agent = agent_class(input_shape, action_dim, config)
    experiment.run_baseline(agent, trial)

    env.close()


def run_rnd_model(env_name, config, i):
    env = create_env(env_name)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentPPO(env_name, env, config)

    agent = PPOAerisRNDAgent(state_dim, action_dim, config)
    experiment.run_rnd_model(agent, i)

    env.close()


def run_dop_model(env_name, config, i):
    env = create_env(env_name)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentPPO(env_name, env, config)

    agent = PPOAerisDOPAgent(state_dim, action_dim, config)
    experiment.run_dop_model(agent, i)

    env.close()


def run_dop_ref_model(env_name, config, i):
    env = create_env(env_name)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentPPO(env_name, env, config)

    agent = PPOAerisDOPRefAgent(state_dim, action_dim, config)
    experiment.run_baseline(agent, i)

    env.close()
