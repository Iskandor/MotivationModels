import gym
import pybullet_envs
import torch
from torch import nn
from torch.nn import *

from algorithms.DDPG import DDPGCritic, DDPGActor, DDPG
from algorithms.ReplayBuffer import ExperienceReplayBuffer
from ddpg_noisy_experiment import ExperimentNoisyDDPG
from modules.NoisyLinear import NoisyLinear
from motivation.ForwardModelMotivation import ForwardModel, ForwardModelMotivation
from motivation.M3Motivation import M3Motivation
from motivation.MateLearnerMotivation import MetaLearnerModel, MetaLearnerMotivation


class Critic(DDPGCritic):
    def __init__(self, state_dim, action_dim, config):
        super(Critic, self).__init__(state_dim, action_dim)

        self._hidden0 = torch.nn.Linear(state_dim, config.critic_h1)
        self._hidden1 = torch.nn.Linear(config.critic_h1 + action_dim, config.critic_h2)
        self._output = torch.nn.Linear(config.critic_h2, 1)

        self.init()

    def forward(self, state, action):
        x = state
        x = torch.tanh(self._hidden0(x))
        x = torch.cat([x, action], 1)
        x = torch.tanh(self._hidden1(x))
        value = self._output(x)
        return value

    def init(self):
        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-3, 3e-3)


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
        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)


class ForwardModelNetwork(ForwardModel):
    def __init__(self, state_dim, action_dim, config):
        super(ForwardModelNetwork, self).__init__(state_dim, action_dim, config)

        self._model = Sequential(
            Linear(in_features=state_dim + action_dim, out_features=config.forward_model_h1, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h1, out_features=config.forward_model_h1, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h1, out_features=config.forward_model_h2, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h2, out_features=config.forward_model_h2, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h2, out_features=state_dim, bias=True)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], state.ndim - 1)
        value = self._model(x)
        return value


class MetaLearnerNetwork(MetaLearnerModel):
    def __init__(self, state_dim, action_dim, config):
        super(MetaLearnerNetwork, self).__init__(state_dim, action_dim, config)
        self._model = Sequential(
            Linear(in_features=state_dim + action_dim, out_features=config.forward_model_h1, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h1, out_features=config.forward_model_h1, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h1, out_features=config.forward_model_h2, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h2, out_features=config.forward_model_h2, bias=True),
            Tanh(),
            Linear(in_features=config.forward_model_h2, out_features=1, bias=True)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], state.ndim - 1)
        value = self._model(x)
        return value


def run_baseline(config):
    env = gym.make('HopperBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentNoisyDDPG('HopperBulletEnv-v0', env, config)

    for i in range(config.trials):
        actor = Actor(state_dim, action_dim, config)
        critic = Critic(state_dim, action_dim, config)
        memory = ExperienceReplayBuffer(config.memory_size)
        agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)
        experiment.run_baseline(agent, i)

    env.close()


def run_forward_model(config):
    env = gym.make('HopperBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentNoisyDDPG('HopperBulletEnv-v0', env, config)

    for i in range(config.trials):
        actor = Actor(state_dim, action_dim, config)
        critic = Critic(state_dim, action_dim, config)
        memory = ExperienceReplayBuffer(config.memory_size)

        agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)

        if hasattr(config, 'forward_model_batch_size'):
            forward_model = ForwardModelMotivation(ForwardModelNetwork(state_dim, action_dim, config), config.forward_model_lr, config.forward_model_eta,
                                                   config.forward_model_variant, env._max_episode_steps * 10,
                                                   memory, config.forward_model_batch_size)
        else:
            forward_model = ForwardModelMotivation(ForwardModelNetwork(state_dim, action_dim, config), config.forward_model_lr, config.forward_model_eta,
                                                   config.forward_model_variant, env._max_episode_steps * 10)

        agent.add_motivation_module(forward_model)

        experiment.run_forward_model(agent, i)

    env.close()


def run_metalearner_model(config):
    env = gym.make('HopperBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentNoisyDDPG('HopperBulletEnv-v0', env, config)

    for i in range(config.trials):
        actor = Actor(state_dim, action_dim, config)
        critic = Critic(state_dim, action_dim, config)
        memory = ExperienceReplayBuffer(config.memory_size)

        agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)

        if hasattr(config, 'forward_model_batch_size'):
            forward_model = ForwardModelMotivation(ForwardModelNetwork(state_dim, action_dim, config), config.forward_model_lr, config.forward_model_eta,
                                                   config.forward_model_variant, env._max_episode_steps * 10,
                                                   memory, config.forward_model_batch_size)
        else:
            forward_model = ForwardModelMotivation(ForwardModelNetwork(state_dim, action_dim, config), config.forward_model_lr, config.forward_model_eta,
                                                   config.forward_model_variant, env._max_episode_steps * 10)

        if hasattr(config, 'metacritic_batch_size'):
            metacritic = MetaLearnerMotivation(MetaLearnerNetwork(state_dim, action_dim, config), forward_model, config.metacritic_lr, state_dim,
                                               config.metacritic_variant, env._max_episode_steps * 10,
                                               config.metacritic_eta, memory, config.metacritic_batch_size)
        else:
            metacritic = MetaLearnerMotivation(MetaLearnerNetwork(state_dim, action_dim, config), forward_model, config.metacritic_lr, state_dim,
                                               config.metacritic_variant, env._max_episode_steps * 10,
                                               config.metacritic_eta)

        agent.add_motivation_module(metacritic)

        experiment.run_metalearner_model(agent, i)

    env.close()