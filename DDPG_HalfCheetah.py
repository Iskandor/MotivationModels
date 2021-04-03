import gym
import pybullet_envs
from algorithms.DDPG import DDPG
from algorithms.ReplayBuffer import ExperienceReplayBuffer
from experiment.ddpg_experiment import ExperimentDDPG
from modules.DDPG_Modules import *
from modules.forward_models.ForwardModel import ForwardModel
from modules.forward_models.ResidualForwardModel import ResidualForwardModel
from modules.forward_models.VAE_ForwardModel import VAE_ForwardModel
from modules.metacritic_models.MetacriticRobotic import MetaCriticRobotic
from motivation.ForwardModelMotivation import ForwardModelMotivation
from motivation.MetaCriticMotivation import MetaCriticMotivation
from motivation.VAE_ForwardModelMotivation import VAE_ForwardModelMotivation


def run_baseline(config, i):
    env = gym.make('HalfCheetahBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('HalfCheetahBulletEnv-v0', env, config)

    actor = Actor(state_dim, action_dim, config)
    critic = Critic(state_dim, action_dim, config)
    memory = ExperienceReplayBuffer(config.memory_size)
    agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)
    experiment.run_baseline(agent, i)

    env.close()


def run_forward_model(config, i):
    env = gym.make('HalfCheetahBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('HalfCheetahBulletEnv-v0', env, config)

    actor = Actor(state_dim, action_dim, config)
    critic = Critic(state_dim, action_dim, config)
    memory = ExperienceReplayBuffer(config.memory_size)

    agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)

    if hasattr(config, 'forward_model_batch_size'):
        forward_model = ForwardModelMotivation(ForwardModel(state_dim, action_dim, config), config.forward_model_lr, config.forward_model_eta,
                                               config.forward_model_variant, env.spec.max_episode_steps * 10,
                                               memory, config.forward_model_batch_size)
    else:
        forward_model = ForwardModelMotivation(ForwardModel(state_dim, action_dim, config), config.forward_model_lr, config.forward_model_eta,
                                               config.forward_model_variant, env.spec.max_episode_steps * 10)

    agent.add_motivation_module(forward_model)

    experiment.run_forward_model(agent, i)

    env.close()


def run_metalearner_model(config, i):
    env = gym.make('HalfCheetahBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('HalfCheetahBulletEnv-v0', env, config)

    actor = Actor(state_dim, action_dim, config)
    critic = Critic(state_dim, action_dim, config)
    memory = ExperienceReplayBuffer(config.memory_size)

    agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)

    if hasattr(config, 'metacritic_batch_size'):
        metacritic = MetaCriticMotivation(MetaCriticRobotic(state_dim, action_dim, config), config.metacritic_lr, config.metacritic_variant, config.metacritic_eta,
                                          memory, config.metacritic_batch_size)
    else:
        metacritic = MetaCriticMotivation(MetaCriticRobotic(state_dim, action_dim, config), config.metacritic_lr, config.metacritic_variant, config.metacritic_eta)

    agent.add_motivation_module(metacritic)

    experiment.run_metalearner_model(agent, i)

    env.close()

def run_residual_forward_model(config, i):
    env = gym.make('HalfCheetahBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('HalfCheetahBulletEnv-v0', env, config)

    actor = Actor(state_dim, action_dim, config)
    critic = Critic(state_dim, action_dim, config)
    memory = ExperienceReplayBuffer(config.memory_size)

    agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)

    forward_model = ForwardModelMotivation(ResidualForwardModel(state_dim, action_dim, config), config.forward_model_lr, config.forward_model_eta,
                                           config.forward_model_variant, env.spec.max_episode_steps * 10,
                                           memory, config.forward_model_batch_size)
    agent.add_motivation_module(forward_model)

    experiment.run_forward_model(agent, i)

    env.close()

def run_vae_forward_model(config, i):
    env = gym.make('HalfCheetahBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('HalfCheetahBulletEnv-v0', env, config)

    actor = Actor(state_dim, action_dim, config)
    critic = Critic(state_dim, action_dim, config)
    memory = ExperienceReplayBuffer(config.memory_size)

    agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)

    forward_model = VAE_ForwardModelMotivation(VAE_ForwardModel(state_dim, action_dim, config), config.forward_model_lr, config.forward_model_eta,
                                               memory, config.forward_model_batch_size)
    agent.add_motivation_module(forward_model)

    experiment.run_vae_forward_model(agent, i)

    env.close()
