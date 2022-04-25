import gym
import torch

from agents import TYPE
from agents.PPOAtariAgent import PPOAtariAgent, PPOAtariRNDAgent, PPOAtariForwardModelAgent, PPOAtariDOPAAgent
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
    if config.model == "baseline":
        experiment = ExperimentPPO(env_name, env, config)
        agent = PPOAtariAgent(input_shape, action_dim, config, TYPE.discrete)
    elif config.model == "dop_a":
        experiment = ExperimentNEnvPPO(env_name, env, config)
        agent = PPOAtariDOPAAgent(input_shape, action_dim, config, TYPE.discrete)
    else:
        raise NotImplementedError(f"Test for the model '{config.model}' is not implemented")
    experiment.add_preprocess(encode_state)

    agent.load(path)
    experiment.test(agent)

    env.close()


def run_baseline(config, trial, env_name):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([WrapperAtari(gym.make(env_name)) for _ in range(config.n_env)], config.n_env, config.num_threads)

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO(env_name, env, config)

    experiment.add_preprocess(encode_state)
    agent = PPOAtariAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_baseline(agent, trial)

    env.close()


def run_rnd_model(config, trial, env_name):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([WrapperAtari(gym.make(env_name)) for _ in range(config.n_env)], config.n_env, config.num_threads)

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO(env_name, env, config)

    experiment.add_preprocess(encode_state)
    agent = PPOAtariRNDAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_rnd_model(agent, trial)

    env.close()


def run_forward_model(config, trial, env_name):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([WrapperAtari(gym.make(env_name)) for _ in range(config.n_env)], config.n_env, config.num_threads)

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO(env_name, env, config)

    experiment.add_preprocess(encode_state)
    agent = PPOAtariForwardModelAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_forward_model(agent, trial)

    env.close()

def run_dop_a_model(config, trial, env_name):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([WrapperAtari(gym.make(env_name)) for _ in range(config.n_env)], config.n_env, config.num_threads)

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO(env_name, env, config)

    experiment.add_preprocess(encode_state)
    agent = PPOAtariDOPAAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_dop_a_model(agent, trial)

    env.close()