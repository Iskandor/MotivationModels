import gym
import pybullet_envs
import torch
from torch import nn

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

        self._hidden0 = nn.Linear(state_dim, config.critic.h1)
        self._hidden1 = nn.Linear(config.critic.h1 + action_dim, config.critic.h2)
        self._output = nn.Linear(config.critic.h2, 1)

        self.init()

    def forward(self, state, action):
        x = state
        x = torch.relu(self._hidden0(x))
        x = torch.cat([x, action], 1)
        x = torch.relu(self._hidden1(x))
        value = self._output(x)
        return value

    def init(self):
        nn.init.xavier_uniform_(self._hidden0.weight)
        nn.init.xavier_uniform_(self._hidden1.weight)
        nn.init.uniform_(self._output.weight, -3e-3, 3e-3)


class Actor(DDPGActor):
    def __init__(self, state_dim, action_dim, config):
        super(Actor, self).__init__(state_dim, action_dim)

        self._hidden0 = nn.Linear(state_dim, config.actor.h1)
        self._hidden1 = nn.Linear(config.actor.h1, config.actor.h2)
        self._output = NoisyLinear(config.actor.h2, action_dim)

        self.init()

    def forward(self, state):
        x = state
        x = torch.relu(self._hidden0(x))
        x = torch.relu(self._hidden1(x))
        policy = torch.tanh(self._output(x))
        return policy

    def init(self):
        nn.init.xavier_uniform_(self._hidden0.weight)
        nn.init.xavier_uniform_(self._hidden1.weight)


class ForwardModelNetwork(ForwardModel):
    def __init__(self, state_dim, action_dim, config):
        super(ForwardModelNetwork, self).__init__(state_dim, action_dim, config)
        self._hidden0 = nn.Linear(state_dim + action_dim, config.forward_model.h1)
        self._hidden1 = nn.Linear(config.forward_model.h1, config.forward_model.h2)
        self._output = nn.Linear(config.forward_model.h2, state_dim)
        self.init()

    def forward(self, state, action):
        x = torch.cat([state, action], state.ndim - 1)
        x = torch.relu(self._hidden0(x))
        x = torch.relu(self._hidden1(x))
        value = self._output(x)
        return value

    def init(self):
        nn.init.xavier_uniform_(self._action.weight)
        nn.init.xavier_uniform_(self._hidden0.weight)
        nn.init.xavier_uniform_(self._hidden1.weight)
        nn.init.uniform_(self._output.weight, -3e-3, 3e-3)


class MetaLearnerNetwork(MetaLearnerModel):
    def __init__(self, state_dim, action_dim, config):
        super(MetaLearnerNetwork, self).__init__(state_dim, action_dim, config)
        self._hidden0 = nn.Linear(state_dim + action_dim, config.metacritic.h1)
        self._hidden1 = nn.Linear(config.metacritic.h1, config.metacritic.h2)
        self._output = nn.Linear(config.metacritic.h2, 1)

        self.init()

    def forward(self, state, action):
        x = torch.cat([state, action], state.ndim - 1)
        x = nn.functional.relu(self._hidden0(x))
        x = nn.functional.relu(self._hidden1(x))
        value = self._output(x)
        return value

    def init(self):
        nn.init.xavier_uniform_(self._hidden0.weight)
        nn.init.xavier_uniform_(self._hidden1.weight)
        nn.init.uniform_(self._output.weight, -3e-1, 3e-1)


class M3Gate(nn.Module):
    def __init__(self, state_dim, im_dim, config):
        super(M3Gate, self).__init__()

        self._hidden0 = nn.Linear(state_dim, config.m3gate.h1)
        self._hidden1 = nn.Linear(config.m3gate.h1, config.m3gate.h2)
        self._output = nn.Linear(config.m3gate.h2, im_dim)

        self.init()

    def forward(self, state):
        x = state
        x = torch.relu(self._hidden0(x))
        x = torch.relu(self._hidden1(x))
        value = self._output(x)
        return value

    def init(self):
        nn.init.xavier_uniform_(self._hidden0.weight)
        nn.init.xavier_uniform_(self._hidden1.weight)
        nn.init.uniform_(self._output.weight, -3e-1, 3e-1)


class M3Critic(nn.Module):
    def __init__(self, state_dim, im_dim, config):
        super(M3Critic, self).__init__()
        self._hidden0 = nn.Linear(state_dim, config.m3critic.h1)
        self._hidden1 = nn.Linear(config.m3critic.h1 + im_dim, config.m3critic.h2)
        self._output = nn.Linear(config.m3critic.h2, 1)
        self.init()

    def forward(self, state, action):
        x = state
        x = torch.relu(self._hidden0(x))
        x = torch.cat([x, action], dim=-1)
        x = torch.relu(self._hidden1(x))
        value = self._output(x)
        return value

    def init(self):
        nn.init.xavier_uniform_(self._hidden0.weight)
        nn.init.xavier_uniform_(self._hidden1.weight)
        nn.init.uniform_(self._output.weight, -3e-1, 3e-1)


def run_baseline(config):
    env = gym.make('HalfCheetahBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentNoisyDDPG('HalfCheetahBulletEnv-v0', env, config)

    for i in range(config.trials):
        actor = Actor(state_dim, action_dim, config)
        critic = Critic(state_dim, action_dim, config)
        memory = ExperienceReplayBuffer(config.memory_size)
        agent = DDPG(actor, critic, config.actor.lr, config.critic.lr, config.gamma, config.tau, memory, config.batch_size)
        experiment.run_baseline(agent, i)

    env.close()


def run_forward_model(config):
    env = gym.make('HalfCheetahBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentNoisyDDPG('HalfCheetahBulletEnv-v0', env, config)

    for i in range(config.trials):
        actor = Actor(state_dim, action_dim, config)
        critic = Critic(state_dim, action_dim, config)
        memory = ExperienceReplayBuffer(config.memory_size)

        agent = DDPG(actor, critic, config.actor.lr, config.critic.lr, config.gamma, config.tau, memory, config.batch_size)

        if config.forward_model.get('batch_size') is not None:
            forward_model = ForwardModelMotivation(ForwardModelNetwork(state_dim, action_dim, config), config.forward_model.lr, config.forward_model.eta,
                                                   config.forward_model.variant, env._max_episode_steps * 10,
                                                   memory, config.forward_model.batch_size)
        else:
            forward_model = ForwardModelMotivation(ForwardModelNetwork(state_dim, action_dim, config), config.forward_model.lr, config.forward_model.eta,
                                                   config.forward_model.variant, env._max_episode_steps * 10)

        agent.add_motivation_module(forward_model)

        experiment.run_forward_model(agent, i)

    env.close()


def run_surprise_model(args):
    args.actor_lr = 1e-4
    args.critic_lr = 2e-4
    args.gamma = 0.99
    args.tau = 1e-3
    args.forward_model_lr = 1e-3
    args.metacritic_lr = 2e-3
    args.eta = 1
    args.metacritic_variant = 'C'

    experiment = ExperimentNoisyDDPG('HalfCheetahBulletEnv-v0', Actor, Critic)
    experiment.run_metalearner_model(args)


def run_metalearner_model(config):
    env = gym.make('HalfCheetahBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentNoisyDDPG('HalfCheetahBulletEnv-v0', env, config)

    for i in range(config.trials):
        actor = Actor(state_dim, action_dim, config)
        critic = Critic(state_dim, action_dim, config)
        memory = ExperienceReplayBuffer(config.memory_size)

        agent = DDPG(actor, critic, config.actor.lr, config.critic.lr, config.gamma, config.tau, memory, config.batch_size)

        if config.forward_model.get('batch_size') is not None:
            forward_model = ForwardModelMotivation(ForwardModelNetwork(state_dim, action_dim, config), config.forward_model.lr, config.forward_model.eta,
                                                   config.forward_model.variant, env._max_episode_steps * 10,
                                                   memory, config.forward_model.batch_size)
        else:
            forward_model = ForwardModelMotivation(ForwardModelNetwork(state_dim, action_dim, config), config.forward_model.lr, config.forward_model.eta,
                                                   config.forward_model.variant, env._max_episode_steps * 10)

        if config.metacritic.get('batch_size') is not None:
            metacritic = MetaLearnerMotivation(MetaLearnerNetwork(state_dim, action_dim, config), forward_model, config.metacritic.lr,
                                               config.metacritic.variant, env._max_episode_steps * 10, config.metacritic.eta,
                                               memory, config.metacritic.batch_size)
        else:
            metacritic = MetaLearnerMotivation(MetaLearnerNetwork(state_dim, action_dim, config), forward_model, config.metacritic.lr,
                                               config.metacritic.variant, env._max_episode_steps * 10, config.metacritic.eta)

        agent.add_motivation_module(metacritic)

        experiment.run_metalearner_model(agent, i)

    env.close()


def run_m3_model(config):
    env = gym.make('HalfCheetahBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentNoisyDDPG('HalfCheetahBulletEnv-v0', env, config)

    for i in range(config.trials):
        actor = Actor(state_dim, action_dim, config)
        critic = Critic(state_dim, action_dim, config)
        agent_memory = ExperienceReplayBuffer(config.memory_size)
        m3_memory = ExperienceReplayBuffer(config.memory_size)

        agent = DDPG(actor, critic, config.actor.lr, config.critic.lr, config.gamma, config.tau, agent_memory, config.batch_size)

        if config.forward_model.get('batch_size') is not None:
            forward_model = ForwardModelMotivation(ForwardModelNetwork(state_dim, action_dim, config), config.forward_model.lr, config.forward_model.eta,
                                                   agent_memory, config.forward_model.batch_size)
        else:
            forward_model = ForwardModelMotivation(ForwardModelNetwork(state_dim, action_dim, config), config.forward_model.lr, config.forward_model.eta)

        if config.metacritic.get('batch_size') is not None:
            metacritic = MetaLearnerMotivation(MetaLearnerNetwork(state_dim, action_dim, config), forward_model, config.metacritic.lr,
                                               config.metacritic.variant, config.metacritic.eta, agent_memory, config.metacritic.batch_size)
        else:
            metacritic = MetaLearnerMotivation(MetaLearnerNetwork(state_dim, action_dim, config), forward_model, config.metacritic.lr,
                                               config.metacritic.variant, config.metacritic.eta)

        m3gate = M3Gate(state_dim * 2, 4, config)
        m3critic = M3Critic(state_dim * 2, 4, config)
        m3module = M3Motivation(m3gate, m3critic, config.m3gate.lr, config.m3critic.lr, config.gamma, config.tau, m3_memory, config.batch_size, forward_model, metacritic)
        agent.add_motivation_module(m3module)

        experiment.run_m3_model(agent, i)

    env.close()
