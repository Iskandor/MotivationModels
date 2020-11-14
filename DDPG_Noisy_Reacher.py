import gym
import torch

from algorithms.DDPG import DDPGCritic, DDPGActor, DDPG
from algorithms.ReplayBuffer import ExperienceReplayBuffer
from ddpg_experiment import ExperimentDDPG
from ddpg_noisy_experiment import ExperimentNoisyDDPG
from modules.NoisyLinear import NoisyLinear
from motivation.ForwardModelMotivation import ForwardModel, ForwardModelMotivation
from motivation.M3Motivation import M3Motivation
from motivation.MateLearnerMotivation import MetaLearnerModel, MetaLearnerMotivation


class Critic(DDPGCritic):
    def __init__(self, state_dim, action_dim, config):
        super(Critic, self).__init__(state_dim, action_dim)

        self._hidden0 = torch.nn.Linear(state_dim, config.critic.h1)
        self._hidden1 = torch.nn.Linear(config.critic.h1 + action_dim, config.critic.h2)
        self._output = torch.nn.Linear(config.critic.h2, 1)

        self.init()

    def forward(self, state, action):
        x = state
        x = torch.relu(self._hidden0(x))
        x = torch.cat([x, action], 1)
        x = torch.relu(self._hidden1(x))
        value = self._output(x)
        return value

    def init(self):
        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-3, 3e-3)


class Actor(DDPGActor):
    def __init__(self, state_dim, action_dim, config):
        super(Actor, self).__init__(state_dim, action_dim)

        self._hidden0 = NoisyLinear(state_dim, config.actor.h1)
        self._hidden1 = NoisyLinear(config.actor.h1, config.actor.h2)
        self._output = NoisyLinear(config.actor.h2, action_dim)

        # self.init()

    def forward(self, state):
        x = state
        x = torch.relu(self._hidden0(x))
        x = torch.relu(self._hidden1(x))
        policy = torch.tanh(self._output(x))
        return policy

    def init(self):
        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)


class ForwardModelNetwork(ForwardModel):
    def __init__(self, state_dim, action_dim, config):
        super(ForwardModelNetwork, self).__init__(state_dim, action_dim, config)
        self._hidden0 = torch.nn.Linear(state_dim + action_dim, config.forward_model.h1)
        self._hidden1 = torch.nn.Linear(config.forward_model.h1, config.forward_model.h2)
        self._output = torch.nn.Linear(config.forward_model.h2, state_dim)
        self.init()

    def forward(self, state, action):
        x = torch.cat([state, action], state.ndim - 1)
        x = torch.relu(self._hidden0(x))
        x = torch.relu(self._hidden1(x))
        value = self._output(x)
        return value

    def init(self):
        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-1, 3e-1)


class MetaLearnerNetwork(MetaLearnerModel):
    def __init__(self, state_dim, action_dim, config):
        super(MetaLearnerNetwork, self).__init__(state_dim, action_dim, config)
        self._hidden0 = torch.nn.Linear(state_dim + action_dim, config.metacritic.h1)
        self._hidden1 = torch.nn.Linear(config.metacritic.h1, config.metacritic.h2)
        self._output = torch.nn.Linear(config.metacritic.h2, 1)

        self.init()

    def forward(self, state, action):
        x = torch.cat([state, action], state.ndim - 1)
        x = torch.nn.functional.relu(self._hidden0(x))
        x = torch.nn.functional.relu(self._hidden1(x))
        value = self._output(x)
        return value

    def init(self):
        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-1, 3e-1)


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
        self._hidden0 = torch.nn.Linear(state_dim, config.m3critic.h1)
        self._hidden1 = torch.nn.Linear(config.m3critic.h1 + im_dim, config.m3critic.h2)
        self._output = torch.nn.Linear(config.m3critic.h2, 1)
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


def run_baseline(config):
    env = gym.make('ReacherPyBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentNoisyDDPG('ReacherPyBulletEnv-v0', env, config)

    for i in range(config.trials):
        actor = Actor(state_dim, action_dim, config)
        critic = Critic(state_dim, action_dim, config)
        memory = ExperienceReplayBuffer(config.memory_size)
        agent = DDPG(actor, critic, config.actor.lr, config.critic.lr, config.gamma, config.tau, memory, config.batch_size)
        experiment.run_baseline(agent, i)

    env.close()


def run_forward_model(config):
    env = gym.make('ReacherPyBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentNoisyDDPG('ReacherPyBulletEnv-v0', env, config)

    for i in range(config.trials):
        actor = Actor(state_dim, action_dim, config)
        critic = Critic(state_dim, action_dim, config)
        memory = ExperienceReplayBuffer(config.memory_size)

        agent = DDPG(actor, critic, config.actor.lr, config.critic.lr, config.gamma, config.tau, memory, config.batch_size)

        if config.forward_model.get('batch_size') is not None:
            forward_model = ForwardModelMotivation(ForwardModelNetwork(state_dim, action_dim, config), config.forward_model.lr, config.forward_model.eta,
                                                   memory, config.forward_model.batch_size)
        else:
            forward_model = ForwardModelMotivation(ForwardModelNetwork(state_dim, action_dim, config), config.forward_model.lr, config.forward_model.eta)

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

    experiment = ExperimentNoisyDDPG('ReacherPyBulletEnv-v0', Actor, Critic)
    experiment.run_metalearner_model(args)


def run_metalearner_model(config):
    env = gym.make('ReacherPyBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentNoisyDDPG('ReacherPyBulletEnv-v0', env, config)

    for i in range(config.trials):
        actor = Actor(state_dim, action_dim, config)
        critic = Critic(state_dim, action_dim, config)
        memory = ExperienceReplayBuffer(config.memory_size)

        agent = DDPG(actor, critic, config.actor.lr, config.critic.lr, config.gamma, config.tau, memory, config.batch_size)

        if config.forward_model.get('batch_size') is not None:
            forward_model = ForwardModelMotivation(ForwardModelNetwork(state_dim, action_dim, config), config.forward_model.lr, config.forward_model.eta,
                                                   memory, config.forward_model.batch_size)
        else:
            forward_model = ForwardModelMotivation(ForwardModelNetwork(state_dim, action_dim, config), config.forward_model.lr, config.forward_model.eta)

        if config.metacritic.get('batch_size') is not None:
            metacritic = MetaLearnerMotivation(MetaLearnerNetwork(state_dim, action_dim, config), forward_model, config.metacritic.lr,
                                               config.metacritic.variant, config.metacritic.eta, memory, config.metacritic.batch_size)
        else:
            metacritic = MetaLearnerMotivation(MetaLearnerNetwork(state_dim, action_dim, config), forward_model, config.metacritic.lr,
                                               config.metacritic.variant, config.metacritic.eta)

        agent.add_motivation_module(metacritic)

        experiment.run_metalearner_model(agent, i)

    env.close()


def run_m3_model(config):
    env = gym.make('ReacherPyBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentNoisyDDPG('ReacherPyBulletEnv-v0', env, config)

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
