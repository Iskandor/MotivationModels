import torch

from agents import TYPE
from agents.PPOProcgenAgent import PPOProcgenCNDAgent, PPOProcgenAgent, PPOProcgenRNDAgent, PPOProcgenQRNDAgent, PPOProcgenSRRNDAgent, PPOProcgenFEDRefAgent, PPOProcgenFWDAgent, PPOProcgenICMAgent
from experiment.ppo_nenv_experiment import ExperimentNEnvPPO
from utils.MultiEnvWrapper import MultiEnvParallel
from utils.ProcgenWrapper import WrapperProcgenExploration


def encode_state(state):
    return torch.tensor(state, dtype=torch.float32)


def test(config, path, env_name):
    env = WrapperProcgenExploration(env_name)
    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    experiment = ExperimentNEnvPPO(env_name, env, config)
    experiment.add_preprocess(encode_state)

    agent = PPOProcgenCNDAgent(input_shape, action_dim, config, TYPE.discrete)
    agent.load(path)
    experiment.test(agent)

    env.close()


def run_baseline(config, trial, env_name):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([WrapperProcgenExploration(env_name) for _ in range(config.n_env)], config.n_env, config.num_threads)

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO(env_name, env, config)

    experiment.add_preprocess(encode_state)
    agent = PPOProcgenAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_baseline(agent, trial)

    env.close()


def run_rnd_model(config, trial, env_name):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([WrapperProcgenExploration(env_name) for _ in range(config.n_env)], config.n_env, config.num_threads)

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO(env_name, env, config)

    experiment.add_preprocess(encode_state)
    agent = PPOProcgenRNDAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_rnd_model(agent, trial)

    env.close()


def run_qrnd_model(config, trial, env_name):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([WrapperProcgenExploration(env_name) for _ in range(config.n_env)], config.n_env, config.num_threads)

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO(env_name, env, config)

    experiment.add_preprocess(encode_state)
    agent = PPOProcgenQRNDAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_qrnd_model(agent, trial)

    env.close()


def run_sr_rnd_model(config, trial, env_name):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([WrapperProcgenExploration(env_name) for _ in range(config.n_env)], config.n_env, config.num_threads)

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO(env_name, env, config)

    experiment.add_preprocess(encode_state)
    agent = PPOProcgenSRRNDAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_sr_rnd_model(agent, trial)

    env.close()


def run_cnd_model(config, trial, env_name):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([WrapperProcgenExploration(env_name) for _ in range(config.n_env)], config.n_env, config.num_threads)

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO(env_name, env, config)

    experiment.add_preprocess(encode_state)
    agent = PPOProcgenCNDAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_cnd_model(agent, trial)

    env.close()


# def run_dop_model(config, trial, env_name):
#     print('Creating {0:d} environments'.format(config.n_env))
#     env = MultiEnvParallel([WrapperProcgenExploration(env_name) for _ in range(config.n_env)], config.n_env, config.num_threads)
#
#     input_shape = env.observation_space.shape
#     action_dim = env.action_space.n
#
#     print('Start training')
#     experiment = ExperimentNEnvPPO(env_name, env, config)
#
#     experiment.add_preprocess(encode_state)
#     agent = PPOProcgenDOPAgent(input_shape, action_dim, config, TYPE.discrete)
#     experiment.run_dop_model(agent, trial)
#
#     env.close()


def run_fed_ref_model(config, trial, env_name):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([WrapperProcgenExploration(env_name) for _ in range(config.n_env)], config.n_env, config.num_threads)

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO(env_name, env, config)

    experiment.add_preprocess(encode_state)
    agent = PPOProcgenFEDRefAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_fed_ref_model(agent, trial)

    env.close()


# def run_forward_model(config, trial, env_name):
#     print('Creating {0:d} environments'.format(config.n_env))
#     env = MultiEnvParallel([WrapperProcgenExploration(env_name) for _ in range(config.n_env)], config.n_env, config.num_threads)
#
#     input_shape = env.observation_space.shape
#     action_dim = env.action_space.n
#
#     print('Start training')
#     experiment = ExperimentNEnvPPO(env_name, env, config)
#
#     experiment.add_preprocess(encode_state)
#     agent = PPOProcgenForwardModelAgent(input_shape, action_dim, config, TYPE.discrete)
#     experiment.run_forward_model(agent, trial)
#
#     env.close()


def run_fwd_model(config, trial, env_name):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([WrapperProcgenExploration(env_name) for _ in range(config.n_env)], config.n_env, config.num_threads)

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO(env_name, env, config)

    experiment.add_preprocess(encode_state)
    agent = PPOProcgenFWDAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_fwd_model(agent, trial)

    env.close()


def run_icm_model(config, trial, env_name):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([WrapperProcgenExploration(env_name) for _ in range(config.n_env)], config.n_env, config.num_threads)

    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO(env_name, env, config)

    experiment.add_preprocess(encode_state)
    agent = PPOProcgenICMAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_icm_model(agent, trial)

    env.close()
