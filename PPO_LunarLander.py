import gym

from agents import TYPE
from agents.PPOAgent import PPOSimpleAgent
from experiment.ppo_experiment import ExperimentPPO
from experiment.ppo_nenv_experiment import ExperimentNEnvPPO


def run_baseline(config, i):
    env = gym.make('LunarLanderContinuous-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if config.n_env > 1:
        env_list = []
        print('Creating {0:d} environments'.format(config.n_env))
        for _ in range(config.n_env):
            env_list.append(gym.make('LunarLanderContinuous-v2'))

        print('Start training')
        experiment = ExperimentNEnvPPO('LunarLanderContinuous-v2', env_list, config)
    else:
        experiment = ExperimentPPO('LunarLanderContinuous-v2', env, config)

    agent = PPOSimpleAgent(state_dim, action_dim, config, TYPE.continuous, config.n_env)
    experiment.run_baseline(agent, i)

    env.close()
