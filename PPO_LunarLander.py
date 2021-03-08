import gym

from agents import TYPE
from agents.PPOAgent import PPOAgent
from experiment.ppo_experiment import ExperimentPPO


def run_baseline(config, i):
    env = gym.make('LunarLanderContinuous-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentPPO('LunarLanderContinuous-v2', env, config)

    agent = PPOAgent(state_dim, action_dim, config, TYPE.continuous)
    experiment.run_baseline(agent, i)

    env.close()
