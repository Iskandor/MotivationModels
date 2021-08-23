import gym

from agents.DQNAgent import DQNSimpleAgent
from experiment.dqn_experiment import ExperimentDQN


def run_baseline(config, i):
    env = gym.make('CartPole-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    experiment = ExperimentDQN('CartPole-v0', env, config)

    agent = DQNSimpleAgent(state_dim, action_dim, config)
    experiment.run_baseline(agent, i)

    env.close()
