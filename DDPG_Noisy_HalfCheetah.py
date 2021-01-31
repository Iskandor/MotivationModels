import gym
import pybullet_envs
import torch
from torch import nn

from algorithms.DDPG import DDPGCritic, DDPGActor, DDPG
from algorithms.ReplayBuffer import ExperienceReplayBuffer
from experiment.ddpg_noisy_experiment import ExperimentNoisyDDPG
from modules.NoisyLinear import NoisyLinear
from modules.forward_models.ForwardModel import ForwardModel
from modules.forward_models.ResidualForwardModel import ResidualForwardModel
from modules.forward_models.VAE_ForwardModel import VAE_ForwardModel
from modules.metacritic_models import MetaCritic
from motivation.ForwardModelMotivation import ForwardModelMotivation
from motivation.MateCriticMotivation import MetaCriticMotivation
from motivation.VAE_ForwardModelMotivation import VAE_ForwardModelMotivation


class Critic(DDPGCritic):
    def __init__(self, state_dim, action_dim, config):
        super(Critic, self).__init__(state_dim, action_dim)

        self._hidden0 = nn.Linear(state_dim, config.critic_h1)
        self._hidden1 = nn.Linear(config.critic_h1 + action_dim, config.critic_h2)
        self._output = nn.Linear(config.critic_h2, 1)

        self.init()

    def forward(self, state, action):
        x = state
        x = torch.tanh(self._hidden0(x))
        x = torch.cat([x, action], 1)
        x = torch.tanh(self._hidden1(x))
        value = self._output(x)
        return value

    def init(self):
        nn.init.xavier_uniform_(self._hidden0.weight)
        nn.init.xavier_uniform_(self._hidden1.weight)
        nn.init.uniform_(self._output.weight, -3e-3, 3e-3)


class Actor(DDPGActor):
    def __init__(self, state_dim, action_dim, config):
        super(Actor, self).__init__(state_dim, action_dim)

        self._hidden0 = nn.Linear(state_dim, config.actor_h1)
        self._hidden1 = nn.Linear(config.actor_h1, config.actor_h2)
        self._output = NoisyLinear(config.actor_h2, action_dim)

        self.init()

    def forward(self, state):
        x = state
        x = torch.tanh(self._hidden0(x))
        x = torch.tanh(self._hidden1(x))
        policy = torch.tanh(self._output(x))
        return policy

    def init(self):
        nn.init.xavier_uniform_(self._hidden0.weight)
        nn.init.xavier_uniform_(self._hidden1.weight)


def run_baseline(config, i):
    env = gym.make('HalfCheetahBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentNoisyDDPG('HalfCheetahBulletEnv-v0', env, config)

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

    experiment = ExperimentNoisyDDPG('HalfCheetahBulletEnv-v0', env, config)

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

    experiment = ExperimentNoisyDDPG('HalfCheetahBulletEnv-v0', env, config)

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

    if hasattr(config, 'metacritic_batch_size'):
        metacritic = MetaCriticMotivation(MetaCritic(state_dim, action_dim, config), forward_model, config.metacritic_lr, state_dim,
                                          config.metacritic_variant, env.spec.max_episode_steps * 10, config.metacritic_eta,
                                          memory, config.metacritic_batch_size)
    else:
        metacritic = MetaCriticMotivation(MetaCritic(state_dim, action_dim, config), forward_model, config.metacritic_lr, state_dim,
                                          config.metacritic_variant, env.spec.max_episode_steps * 10, config.metacritic_eta)

    agent.add_motivation_module(metacritic)

    experiment.run_metalearner_model(agent, i)

    env.close()


def run_residual_forward_model(config, i):
    env = gym.make('HalfCheetahBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentNoisyDDPG('HalfCheetahBulletEnv-v0', env, config)

    actor = Actor(state_dim, action_dim, config)
    critic = Critic(state_dim, action_dim, config)
    memory = ExperienceReplayBuffer(config.memory_size)

    agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)

    forward_model = ForwardModelMotivation(ResidualForwardModel(state_dim, action_dim, config), config.forward_model_lr,
                                           config.forward_model_eta,
                                           config.forward_model_variant, env.spec.max_episode_steps * 10,
                                           memory, config.forward_model_batch_size)
    agent.add_motivation_module(forward_model)

    experiment.run_forward_model(agent, i)

    env.close()


def run_vae_forward_model(config, i):
    env = gym.make('HalfCheetahBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentNoisyDDPG('HalfCheetahBulletEnv-v0', env, config)

    actor = Actor(state_dim, action_dim, config)
    critic = Critic(state_dim, action_dim, config)
    memory = ExperienceReplayBuffer(config.memory_size)

    agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)

    forward_model = VAE_ForwardModelMotivation(VAE_ForwardModel(state_dim, action_dim, config), config.forward_model_lr,
                                               config.forward_model_eta,
                                               memory, config.forward_model_batch_size)
    agent.add_motivation_module(forward_model)

    experiment.run_vae_forward_model(agent, i)

    env.close()
