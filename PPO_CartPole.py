import gym

from agents import TYPE
from agents.PPOAgent import PPOSimpleAgent
from experiment.ppo_experiment import ExperimentPPO
from experiment.ppo_nenv_experiment import ExperimentNEnvPPO


def run_baseline(config, trial):
    env = gym.make('CartPole-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if config.n_env > 1:
        env_list = []
        print('Creating {0:d} environments'.format(config.n_env))
        for i in range(config.n_env):
            env_list.append(gym.make('CartPole-v0'))

        print('Start training')
        experiment = ExperimentNEnvPPO('Pitfall-v0', env_list, config)
    else:
        experiment = ExperimentPPO('Pitfall-v0', env, config)

    agent = PPOSimpleAgent(state_dim, action_dim, config, TYPE.discrete)
    experiment.run_baseline(agent, trial)

    env.close()

    if config.n_env > 1:
        for i in range(config.n_env):
            env_list[i].close()
