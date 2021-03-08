import gym

from agents import TYPE
from agents.PPOAgent import PPOAgent
from experiment.ppo_experiment import ExperimentPPO


def run_baseline(config, i):
    env = gym.make('CartPole-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    experiment = ExperimentPPO('CartPole-v0', env, config)

    agent = PPOAgent(state_dim, action_dim, config, TYPE.discrete)
    experiment.run_baseline(agent, i)

    env.close()
