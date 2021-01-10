import gym

from algorithms.DDPG import DDPG
from algorithms.ReplayBuffer import ExperienceReplayBuffer
from ddpg_experiment import ExperimentDDPG
from modules.DDPG_Modules import *
from motivation.ForwardModelMotivation import ForwardModelMotivation
from motivation.MateLearnerMotivation import MetaLearnerMotivation


def run_baseline(config, i):
    env = gym.make('LunarLanderContinuous-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('LunarLanderContinuous-v2', env, config)

    actor = Actor(state_dim, action_dim, config)
    critic = Critic(state_dim, action_dim, config)
    memory = ExperienceReplayBuffer(config.memory_size)
    agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)
    experiment.run_baseline(agent, i)

    env.close()


def run_forward_model(config, i):
    env = gym.make('LunarLanderContinuous-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('LunarLanderContinuous-v2', env, config)

    actor = Actor(state_dim, action_dim, config)
    critic = Critic(state_dim, action_dim, config)
    memory = ExperienceReplayBuffer(config.memory_size)

    agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)

    if hasattr(config, 'forward_model_batch_size'):
        forward_model = ForwardModelMotivation(SmallForwardModelNetwork(state_dim, action_dim, config), config.forward_model_lr, config.forward_model_eta,
                                               memory, config.forward_model_batch_size)
    else:
        forward_model = ForwardModelMotivation(SmallForwardModelNetwork(state_dim, action_dim, config), config.forward_model_lr, config.forward_model_eta)

    agent.add_motivation_module(forward_model)

    experiment.run_forward_model(agent, i)

    env.close()


def run_metalearner_model(config, i):
    env = gym.make('LunarLanderContinuous-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('LunarLanderContinuous-v2', env, config)

    actor = Actor(state_dim, action_dim, config)
    critic = Critic(state_dim, action_dim, config)
    memory = ExperienceReplayBuffer(config.memory_size)

    agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)

    if hasattr(config, 'forward_model_batch_size'):
        forward_model = ForwardModelMotivation(SmallForwardModelNetwork(state_dim, action_dim, config), config.forward_model_lr, config.forward_model_eta,
                                               memory, config.forward_model_batch_size)
    else:
        forward_model = ForwardModelMotivation(SmallForwardModelNetwork(state_dim, action_dim, config), config.forward_model_lr, config.forward_model_eta)

    if hasattr(config, 'metacritic_batch_size'):
        metacritic = MetaLearnerMotivation(SmallMetaLearnerNetwork(state_dim, action_dim, config), forward_model, config.metacritic_lr, state_dim,
                                           config.metacritic_variant, config.metacritic_eta, memory, config.metacritic_batch_size)
    else:
        metacritic = MetaLearnerMotivation(SmallMetaLearnerNetwork(state_dim, action_dim, config), forward_model, config.metacritic_lr, state_dim,
                                           config.metacritic_variant, config.metacritic_eta)

    agent.add_motivation_module(metacritic)

    experiment.run_metalearner_model(agent, i)

    env.close()
