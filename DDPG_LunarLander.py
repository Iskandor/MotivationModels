import gym

from agents.DDPGAgent import DDPGAgent
from experiment.ddpg_experiment import ExperimentDDPG


def run_baseline(config, i):
    env = gym.make('LunarLanderContinuous-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('LunarLanderContinuous-v2', env, config)

    agent = DDPGAgent(state_dim, action_dim, config)
    experiment.run_baseline(agent, i)

    env.close()


def run_forward_model(config, i):
    pass


def run_metalearner_model(config, i):
    pass
