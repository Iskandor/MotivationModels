import gym
import torch
from torch import nn

from algorithms.DDPG import DDPGCritic, DDPGActor, DDPG
from algorithms.ReplayBuffer import ExperienceReplayBuffer
from experiment.ddpg_noisy_experiment import ExperimentNoisyDDPG
from modules.NoisyLinear import NoisyLinear
from modules.forward_models.ForwardModel import ForwardModel
from modules.metacritic_models import MetaCritic
from motivation.ForwardModelMotivation import ForwardModelMotivation
from motivation.M3Motivation import M3Motivation
from motivation.MateCriticMotivation import MetaCriticMotivation


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


class M3Gate(torch.nn.Module):
    def __init__(self, state_dim, im_dim, config):
        super(M3Gate, self).__init__()

        self._hidden0 = torch.nn.Linear(state_dim, config.m3gate.h1)
        self._hidden1 = torch.nn.Linear(config.m3gate.h1, config.m3gate.h2)
        self._output = torch.nn.Linear(config.m3gate.h2, im_dim)

        self.init()

    def forward(self, state):
        x = state
        x = torch.relu(self._hidden0(x))
        x = torch.relu(self._hidden1(x))
        value = self._output(x)
        return value

    def init(self):
        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-1, 3e-1)


class M3Critic(torch.nn.Module):
    def __init__(self, state_dim, im_dim, config):
        super(M3Critic, self).__init__()
        self._hidden0 = torch.nn.Linear(state_dim, config.m3critic_h1)
        self._hidden1 = torch.nn.Linear(config.m3critic_h1 + im_dim, config.m3critic_h2)
        self._output = torch.nn.Linear(config.m3critic_h2, 1)
        self.init()

    def forward(self, state, action):
        x = state
        x = torch.relu(self._hidden0(x))
        x = torch.cat([x, action], dim=-1)
        x = torch.relu(self._hidden1(x))
        value = self._output(x)
        return value

    def init(self):
        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-1, 3e-1)


def run_baseline(config, i):
    env = gym.make('ReacherBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentNoisyDDPG('ReacherBulletEnv-v0', env, config)

    actor = Actor(state_dim, action_dim, config)
    critic = Critic(state_dim, action_dim, config)
    memory = ExperienceReplayBuffer(config.memory_size)
    agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)
    experiment.run_baseline(agent, i)

    env.close()


def run_forward_model(config, i):
    env = gym.make('ReacherBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentNoisyDDPG('ReacherBulletEnv-v0', env, config)

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
    env = gym.make('ReacherBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentNoisyDDPG('ReacherBulletEnv-v0', env, config)

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
                                          config.metacritic_variant, env.spec.max_episode_steps * 10,
                                          config.metacritic_eta, memory, config.metacritic_batch_size)
    else:
        metacritic = MetaCriticMotivation(MetaCritic(state_dim, action_dim, config), forward_model, config.metacritic_lr, state_dim,
                                          config.metacritic_variant, env.spec.max_episode_steps * 10,
                                          config.metacritic_eta)

    agent.add_motivation_module(metacritic)

    experiment.run_metalearner_model(agent, i)

    env.close()


def run_m3_model(config, i):
    env = gym.make('ReacherBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentNoisyDDPG('ReacherBulletEnv-v0', env, config)

    actor = Actor(state_dim, action_dim, config)
    critic = Critic(state_dim, action_dim, config)
    agent_memory = ExperienceReplayBuffer(config.memory_size)
    m3_memory = ExperienceReplayBuffer(config.memory_size)

    agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, agent_memory, config.batch_size)

    if hasattr(config, 'forward_model_batch_size'):
        forward_model = ForwardModelMotivation(ForwardModel(state_dim, action_dim, config), config.forward_model_lr, config.forward_model_eta,
                                               agent_memory, config.forward_model_batch_size)
    else:
        forward_model = ForwardModelMotivation(ForwardModel(state_dim, action_dim, config), config.forward_model_lr, config.forward_model_eta)

    if hasattr(config, 'metacritic_batch_size'):
        metacritic = MetaCriticMotivation(MetaCritic(state_dim, action_dim, config), forward_model, config.metacritic_lr,
                                          config.metacritic_variant, config.metacritic_eta, agent_memory, config.metacritic_batch_size)
    else:
        metacritic = MetaCriticMotivation(MetaCritic(state_dim, action_dim, config), forward_model, config.metacritic_lr,
                                          config.metacritic_variant, config.metacritic_eta)

    m3gate = M3Gate(state_dim * 2, 4, config)
    m3critic = M3Critic(state_dim * 2, 4, config)
    m3module = M3Motivation(m3gate, m3critic, config.m3gate.lr, config.m3critic_lr, config.gamma, config.tau, m3_memory, config.batch_size, forward_model, metacritic)
    agent.add_motivation_module(m3module)

    experiment.run_m3_model(agent, i)

    env.close()
