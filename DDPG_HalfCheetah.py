import gym
import torch

from algorithms.DDPG import DDPGCritic, DDPGActor, DDPG
from algorithms.ReplayBuffer import ExperienceReplayBuffer
from ddpg_experiment import ExperimentDDPG
from motivation.ForwardModelMotivation import ForwardModel, ForwardModelMotivation
from motivation.MateLearnerMotivation import MetaLearnerModel


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

        self._hidden0 = torch.nn.Linear(state_dim, config.actor.h1)
        self._hidden1 = torch.nn.Linear(config.actor.h1, config.actor.h2)
        self._output = torch.nn.Linear(config.actor.h2, action_dim)

        self.init()

    def forward(self, state):
        x = state
        x = torch.relu(self._hidden0(x))
        x = torch.relu(self._hidden1(x))
        policy = torch.tanh(self._output(x))
        return policy

    def init(self):
        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-1, 3e-1)


class ForwardModelNetwork(ForwardModel):
    def __init__(self, state_dim, action_dim, config):
        super(ForwardModelNetwork, self).__init__(state_dim, action_dim)
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
    def __init__(self, state_dim, action_dim):
        super(MetaLearnerNetwork, self).__init__(state_dim, action_dim)
        self._hidden0 = torch.nn.Linear(state_dim + action_dim, 128)
        self._hidden1 = torch.nn.Linear(128, 64)
        self._output = torch.nn.Linear(64, 1)

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


def run_baseline(config):
    env = gym.make('HalfCheetahPyBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('HalfCheetahPyBulletEnv-v0', env, config)

    for i in range(config.trials):
        actor = Actor(state_dim, action_dim, config)
        critic = Critic(state_dim, action_dim, config)
        memory = ExperienceReplayBuffer(config.memory_size)
        agent = DDPG(actor, critic, config.actor.lr, config.critic.lr, config.gamma, config.tau, memory, config.batch_size)
        experiment.run_baseline(agent, i)

    env.close()


def run_forward_model(config):
    env = gym.make('HalfCheetahPyBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('HalfCheetahPyBulletEnv-v0', env, config)

    for i in range(config.trials):
        actor = Actor(state_dim, action_dim, config)
        critic = Critic(state_dim, action_dim, config)
        memory = ExperienceReplayBuffer(config.memory_size)
        forward_model = ForwardModelNetwork(state_dim, action_dim, config)
        motivation = ForwardModelMotivation(forward_model, config.forward_model.lr, config.forward_model.eta)
        agent = DDPG(actor, critic, config.actor.lr, config.critic.lr, config.gamma, config.tau, memory, config.batch_size)
        agent.add_motivation_module(motivation)
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

    experiment = ExperimentDDPG('HalfCheetahPyBulletEnv-v0', Actor, Critic, ForwardModelNetwork, MetaLearnerNetwork)
    experiment.run_metalearner_model(args)


def run_metalearner_model(args):
    args.actor_lr = 1e-4
    args.critic_lr = 2e-4
    args.gamma = 0.99
    args.tau = 1e-3
    args.forward_model_lr = 1e-3
    args.metacritic_lr = 2e-3
    args.eta = 1
    args.metacritic_variant = 'A'

    experiment = ExperimentDDPG('HalfCheetahPyBulletEnv-v0', Actor, Critic, ForwardModelNetwork, MetaLearnerNetwork)
    experiment.run_metalearner_model(args)
