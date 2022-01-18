import gym
import torch

from agents import TYPE
from agents.PPOAtariAgent import PPOAtariAgent, PPOAtariRNDAgent, PPOAtariForwardModelAgent, PPOAtariQRNDAgent, PPOAtariDOPAgent, PPOAtariSRRNDAgent, PPOAtariDOPAgent2
from experiment.ppo_nenv_experiment import ExperimentNEnvPPO
from utils.AtariWrapper import WrapperHardAtari
from utils.MultiEnvWrapper import MultiEnvParallel


def encode_state(state):
    return torch.tensor(state, dtype=torch.float32)


def test(config, path, env_name):
    env = WrapperHardAtari(gym.make(env_name))
    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    experiment = ExperimentNEnvPPO(env_name, env, config)
    experiment.add_preprocess(encode_state)

    agent = PPOAtariDOPAgent(input_shape, action_dim, config, TYPE.discrete)
    agent.load(path)
    experiment.test(agent)

    env.close()


def run_baseline(config, trial, env_name):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([WrapperHardAtari(gym.make(env_name)) for _ in range(config.n_env)], config.n_env, config.num_threads)

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
    env = MultiEnvParallel([WrapperHardAtari(gym.make(env_name)) for _ in range(config.n_env)], config.n_env, config.num_threads)

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO(env_name, env, config)

    experiment.add_preprocess(encode_state)
    agent = PPOAtariRNDAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_rnd_model(agent, trial)

    env.close()


def run_qrnd_model(config, trial, env_name):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([WrapperHardAtari(gym.make(env_name)) for _ in range(config.n_env)], config.n_env, config.num_threads)

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO(env_name, env, config)

    experiment.add_preprocess(encode_state)
    agent = PPOAtariQRNDAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_qrnd_model(agent, trial)

    env.close()

def run_sr_rnd_model(config, trial, env_name):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([WrapperHardAtari(gym.make(env_name)) for _ in range(config.n_env)], config.n_env, config.num_threads)

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO(env_name, env, config)

    experiment.add_preprocess(encode_state)
    agent = PPOAtariSRRNDAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_sr_rnd_model(agent, trial)

    env.close()


def run_dop_model(config, trial, env_name):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([WrapperHardAtari(gym.make(env_name)) for _ in range(config.n_env)], config.n_env, config.num_threads)

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO(env_name, env, config)

    experiment.add_preprocess(encode_state)
    agent = PPOAtariDOPAgent2(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_dop_model(agent, trial)

    env.close()


def run_forward_model(config, trial, env_name):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([WrapperHardAtari(gym.make(env_name)) for _ in range(config.n_env)], config.n_env, config.num_threads)

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO(env_name, env, config)

    experiment.add_preprocess(encode_state)
    agent = PPOAtariForwardModelAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_forward_model(agent, trial)

    env.close()
