import gym

from agents import TYPE
from agents.PPOAgent import PPOSimpleAgent
from experiment.ppo_experiment import ExperimentPPO
from experiment.ppo_nenv_experiment import ExperimentNEnvPPO
from utils.MultiEnvWrapper import MultiEnvParallel


def run_baseline(config, trial):
    # if config.n_env > 1:
    #     print('Creating {0:d} environments'.format(config.n_env))
    #     env = MultiEnvParallel([gym.make('gym_bitflip:bitflip-v0') for _ in range(config.n_env)], config.n_env, config.num_threads)
    # else:
    #     env = gym.make('gym_bitflip:bitflip-v0')

    print('Creating {0:d} environments'.format(config.n_env))
    env = MultiEnvParallel([gym.make('gym_bitflip:bitflip-v0') for _ in range(config.n_env)], config.n_env, config.num_threads)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print('Start training')
    # if config.n_env > 1:
    #     experiment = ExperimentNEnvPPO('bitflip-v0', env, config)
    # else:
    #     experiment = ExperimentPPO('bitflip-v0', env, config)
    experiment = ExperimentNEnvPPO('bitflip-v0', env, config)
    agent = PPOSimpleAgent(state_dim, action_dim, config, TYPE.discrete)
    experiment.run_baseline(agent, trial)

    env.close()
