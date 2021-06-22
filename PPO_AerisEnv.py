import gym
import gym_aeris.envs

from agents import TYPE
from agents.PPOAgent import PPOAerisAgent, PPOAerisRNDAgent, PPOAerisDOPSimpleAgent
from experiment.ppo_experiment import ExperimentPPO


def run_baseline(env_name, env, config, i):
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    env_list = None
    if config.n_env > 1:
        env_list = []
        print('Creating {0:d} environments'.format(config.n_env))
        for _ in range(config.n_env):
            env_list.append(WrapperAtari(gym.make(env_name)))

        print('Start training')
        experiment = ExperimentNEnvPPO(env_name, env_list, config, state_dim, action_dim)
    else:
        experiment = ExperimentPPO(env_name, env, config)
        experiment.add_preprocess(encode_state)

    agent = PPOAerisAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_baseline(agent, trial)

    env.close()

    if config.n_env > 1:
        for i in range(config.n_env):
            env_list[i].close()


def run_rnd_model(env_name, env, config, i):
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentPPO(env_name, env, config)

    agent = PPOAerisRNDAgent(state_dim, action_dim, config, TYPE.continuous)
    experiment.run_rnd_model(agent, i)

    env.close()


def run_dop_model(env_name, env, config, i):
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentPPO(env_name, env, config)

    agent = PPOAerisDOPSimpleAgent(state_dim, action_dim, config, TYPE.continuous)
    experiment.run_dop_model(agent, i)

    env.close()
