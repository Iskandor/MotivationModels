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


def run_env(env_name, config, trial, agent_class, experiment_type):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([create_env(env_name) for _ in range(config.n_env)], config.n_env, config.num_threads)

    input_shape = env.observation_space.shape
    if isinstance(env.action_space, Box):
        action_dim = env.action_space.shape[0]
    if isinstance(env.action_space, Discrete):
        action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO(env_name, env, config)
    method_to_call = getattr(experiment, experiment_type)

    agent = agent_class(input_shape, action_dim, config)
    method_to_call(agent, trial)

    env.close()


def run_baseline(env_name, config, trial, agent_class):
    run_env(env_name, config, trial, agent_class, 'run_baseline')


def run_rnd_model(env_name, config, trial, agent_class):
    run_env(env_name, config, trial, agent_class, 'run_rnd_model')


def run_dop_model(env_name, config, trial, agent_class):
    run_env(env_name, config, trial, agent_class, 'run_dop_model')


def run_dop_ref_model(env_name, config, trial, agent_class):
    run_env(env_name, config, trial, agent_class, 'run_baseline')
