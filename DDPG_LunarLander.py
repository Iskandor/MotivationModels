import gym

from algorithms.DDPG import DDPG
from algorithms.ReplayBuffer import ExperienceReplayBuffer
from experiment.ddpg_experiment import ExperimentDDPG
from modules import forward_models, metacritic_models, ARCH
from modules.DDPG_Modules import *
from modules.M2_Modules import M2Gate
from modules.forward_models.ForwardModel import ForwardModel
from modules.metacritic_models import MetaCritic
from motivation.ForwardModelMotivation import ForwardModelMotivation
from motivation.M2Motivation import M2Motivation
from motivation.MateCriticMotivation import MetaCriticMotivation


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
        forward_model = ForwardModelMotivation(ForwardModel(state_dim, action_dim, config, ARCH.robotic), config.forward_model_lr,
                                               config.forward_model_eta, config.forward_model_variant, 0,
                                               memory, config.forward_model_batch_size)
    else:
        forward_model = ForwardModelMotivation(ForwardModel(state_dim, action_dim, config, ARCH.robotic), config.forward_model_lr,
                                               config.forward_model_eta, config.forward_model_variant)

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

    if hasattr(config, 'metacritic_batch_size'):
        metacritic = MetaCriticMotivation(MetaCritic(state_dim, action_dim, config), config.metacritic_lr, config.metacritic_variant, config.metacritic_eta,
                                          memory, config.metacritic_batch_size)
    else:
        metacritic = MetaCriticMotivation(MetaCritic(state_dim, action_dim, config), config.metacritic_lr, config.metacritic_variant, config.metacritic_eta)

    agent.add_motivation_module(metacritic)

    experiment.run_metalearner_model(agent, i)

    env.close()

def run_m2_model(config, i):
    env = gym.make('LunarLanderContinuous-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('LunarLanderContinuous-v2', env, config)

    actor = Actor(state_dim, action_dim, config)
    critic = Critic(state_dim, action_dim, config)
    memory = ExperienceReplayBuffer(config.memory_size)
    m2_memory = ExperienceReplayBuffer(config.memory_size)

    agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)

    if hasattr(config, 'forward_model_batch_size'):
        forward_model = ForwardModelMotivation(ForwardModel(state_dim, action_dim, config, ARCH.robotic), config.forward_model_lr,
                                               config.forward_model_eta, config.forward_model_variant, 0,
                                               memory, config.forward_model_batch_size)
    else:
        forward_model = ForwardModelMotivation(ForwardModel(state_dim, action_dim, config, ARCH.robotic), config.forward_model_lr,
                                               config.forward_model_eta, config.forward_model_variant)

    m2_model = M2Motivation(M2Gate(state_dim * 2, 2, config), 1e-4, config.gamma, config.tau, m2_memory, config.batch_size, actor, critic, forward_model)

    agent.add_motivation_module(m2_model)

    experiment.run_m2_model(agent, i)

    env.close()
