import gym
import torch

from agents import TYPE
from agents.PPOAgent import PPOAtariAgent, PPOAtariForwardModelAgent, PPOAtariRNDAgent
from experiment.ppo_experiment import ExperimentPPO
from experiment.ppo_nenv_experiment import ExperimentNEnvPPO
from utils.AtariWrapper import WrapperAtari
from utils.MultiEnvWrapper import MultiEnvParallel


def encode_state(state):
    return torch.tensor(state, dtype=torch.float32)


def test(config, path, env_name):
    env = WrapperAtari(gym.make(env_name))
    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    experiment = ExperimentPPO(env_name, env, config)
    experiment.add_preprocess(encode_state)

    agent = PPOAtariAgent(input_shape, action_dim, config, TYPE.discrete)
    agent.load(path)
    experiment.test(agent)

    env.close()


def run_baseline(config, trial, env_name):
    if config.n_env > 1:
        print('Creating {0:d} environments'.format(config.n_env))
        env = MultiEnvParallel([WrapperAtari(gym.make(env_name)) for _ in range(config.n_env)], config.n_env, config.num_threads)
    else:
        env = WrapperAtari(gym.make(env_name))

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print('Start training')
    if config.n_env > 1:
        experiment = ExperimentNEnvPPO(env_name, env, config)
    else:
        experiment = ExperimentPPO(env_name, env, config)

    experiment.add_preprocess(encode_state)
    agent = PPOAtariAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_baseline(agent, trial)

    env.close()


def run_rnd_model(config, trial, env_name):
    if config.n_env > 1:
        print('Creating {0:d} environments'.format(config.n_env))
        env = MultiEnvParallel([WrapperAtari(gym.make(env_name)) for _ in range(config.n_env)], config.n_env, config.num_threads)
    else:
        env = WrapperAtari(gym.make(env_name))

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print('Start training')
    if config.n_env > 1:
        experiment = ExperimentNEnvPPO(env_name, env, config)
    else:
        experiment = ExperimentPPO(env_name, env, config)

    experiment.add_preprocess(encode_state)
    agent = PPOAtariRNDAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_rnd_model(agent, trial)

    env.close()


def run_forward_model(config, trial, env_name):
    env = WrapperAtari(gym.make(env_name))
    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    if config.n_env > 1:
        env_list = []
        print('Creating {0:d} environments'.format(config.n_env))
        for i in range(config.n_env):
            env_list.append(WrapperAtari(gym.make(env_name)))

        print('Start training')
        experiment = ExperimentNEnvPPO(env_name, env_list, config, input_shape, action_dim)
    else:
        experiment = ExperimentPPO(env_name, env, config)
        experiment.add_preprocess(encode_state)

    agent = PPOAtariForwardModelAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_forward_model(agent, trial)

    env.close()

    if config.n_env > 1:
        for i in range(config.n_env):
            env_list[i].close()
