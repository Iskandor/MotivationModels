import gym

from modules import ARCH
from modules.DDPG_Modules import *
from algorithms.ReplayBuffer import ExperienceReplayBuffer
from experiment.ddpg_experiment import ExperimentDDPG
from algorithms.DDPG import DDPG
from modules.forward_models.ForwardModel import ForwardModel
from modules.metacritic_models import SmallMetaCritic
from motivation.ForwardModelMotivation import ForwardModelMotivation
from motivation.MateCriticMotivation import MetaCriticMotivation


def run_baseline(config, i):
    env = gym.make('MountainCarContinuous-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('MountainCarContinuous-v0', env, config)

    actor = Actor(state_dim, action_dim, config)
    critic = Critic(state_dim, action_dim, config)
    memory = ExperienceReplayBuffer(config.memory_size)
    agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)
    experiment.run_baseline(agent, i)

    env.close()


def run_forward_model(config, i):
    env = gym.make('MountainCarContinuous-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('MountainCarContinuous-v0', env, config)

    actor = Actor(state_dim, action_dim, config)
    critic = Critic(state_dim, action_dim, config)
    memory = ExperienceReplayBuffer(config.memory_size)

    agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)

    if hasattr(config, 'forward_model_batch_size'):
        forward_model = ForwardModelMotivation(ForwardModel(state_dim, action_dim, config, ARCH.small_robotic), config.forward_model_lr, config.forward_model_eta,
                                               config.forward_model_variant, env.spec.max_episode_steps * 10,
                                               memory, config.forward_model_batch_size)
    else:
        forward_model = ForwardModelMotivation(ForwardModel(state_dim, action_dim, config, ARCH.small_robotic), config.forward_model_lr, config.forward_model_eta,
                                               config.forward_model_variant, env.spec.max_episode_steps * 10)

    agent.add_motivation_module(forward_model)

    experiment.run_forward_model(agent, i)

    env.close()


def run_metalearner_model(config, i):
    env = gym.make('MountainCarContinuous-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('MountainCarContinuous-v0', env, config)

    actor = Actor(state_dim, action_dim, config)
    critic = Critic(state_dim, action_dim, config)
    memory = ExperienceReplayBuffer(config.memory_size)

    agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)

    if hasattr(config, 'forward_model_batch_size'):
        forward_model = ForwardModelMotivation(ForwardModel(state_dim, action_dim, config, ARCH.small_robotic), config.forward_model_lr, config.forward_model_eta,
                                               config.forward_model_variant, env.spec.max_episode_steps * 10,
                                               memory, config.forward_model_batch_size)
    else:
        forward_model = ForwardModelMotivation(ForwardModel(state_dim, action_dim, config, ARCH.small_robotic), config.forward_model_lr, config.forward_model_eta,
                                               config.forward_model_variant, env.spec.max_episode_steps * 10)

    if hasattr(config, 'metacritic_batch_size'):
        metacritic = MetaCriticMotivation(SmallMetaCritic(state_dim, action_dim, config), forward_model, config.metacritic_lr, state_dim,
                                          config.metacritic_variant, env.spec.max_episode_steps * 10, config.metacritic_eta,
                                          memory, config.metacritic_batch_size)
    else:
        metacritic = MetaCriticMotivation(SmallMetaCritic(state_dim, action_dim, config), forward_model, config.metacritic_lr, state_dim,
                                          config.metacritic_variant, env.spec.max_episode_steps * 10, config.metacritic_eta)

    agent.add_motivation_module(metacritic)

    experiment.run_metalearner_model(agent, i)

    env.close()
