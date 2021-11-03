import gym

from agents import TYPE
from agents.PPOBitFlipAgent import PPOBitFlipAgent, PPOBitFlipRNDAgent, PPOBitFlipQRNDAgent, PPOBitFlipDOPAgent
from experiment.ppo_nenv_experiment import ExperimentNEnvPPO
from utils.MultiEnvWrapper import MultiEnvParallel

dimension = 8

def run_baseline(config, trial):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([gym.make('gym_bitflip:bitflip-v0', dimension=dimension) for _ in range(config.n_env)], config.n_env, config.num_threads)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO('bitflip-v0', env, config)
    agent = PPOBitFlipAgent(state_dim, action_dim, config, TYPE.discrete)
    experiment.run_baseline(agent, trial)

    env.close()


def run_rnd_model(config, trial):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([gym.make('gym_bitflip:bitflip-v0', dimension=dimension) for _ in range(config.n_env)], config.n_env, config.num_threads)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO('bitflip-v0', env, config)
    agent = PPOBitFlipRNDAgent(state_dim, action_dim, config, TYPE.discrete)
    experiment.run_rnd_model(agent, trial)

    env.close()


def run_qrnd_model(config, trial):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([gym.make('gym_bitflip:bitflip-v0', dimension=dimension) for _ in range(config.n_env)], config.n_env, config.num_threads)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO('bitflip-v0', env, config)
    agent = PPOBitFlipQRNDAgent(state_dim, action_dim, config, TYPE.discrete)
    experiment.run_qrnd_model(agent, trial)

    env.close()


def run_dop_model(config, trial):
    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([gym.make('gym_bitflip:bitflip-v0', dimension=dimension) for _ in range(config.n_env)], config.n_env, config.num_threads)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print('Start training')
    experiment = ExperimentNEnvPPO('bitflip-v0', env, config)
    agent = PPOBitFlipDOPAgent(state_dim, action_dim, config, TYPE.discrete)
    experiment.run_dop_model(agent, trial)

    env.close()
