import gym
import gym_aeris
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
    def __init__(self, input_shape, action_dim, config):
        super(Critic, self).__init__(input_shape, action_dim)

        self.channels = input_shape[0]
        self.width = input_shape[1]

        fc_count = config.critic_kernels_count * self.width // 4

        self._network = nn.Sequential(
            nn.Conv1d(self.channels + action_dim, config.critic_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_count, config.critic_h1),
            nn.ReLU(),
            nn.Linear(config.critic_h1, 1))

    def forward(self, state, action):
        a = action.unsqueeze(2).repeat(1, 1, state.shape[2])
        x = torch.cat([state, a], dim=1)
        value = self._network(x)
        return value


class Actor(DDPGActor):
    def __init__(self, input_shape, action_dim, config):
        super(Actor, self).__init__(input_shape, action_dim)

        self.channels = input_shape[0]
        self.width = input_shape[1]

        fc_count = config.actor_kernels_count * self.width // 4

        self._network = nn.Sequential(
            nn.Conv1d(self.channels, config.actor_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_count, config.actor_h1),
            nn.ReLU(),
            NoisyLinear(config.actor_h1, action_dim),
            nn.Tanh())

    def forward(self, state):
        if state.ndim == 2:
            state = state.unsqueeze(0)
            policy = self._network(state).squeeze(0)
        else:
            policy = self._network(state)
        return policy


class ForwardModelNetwork(ForwardModel):
    def __init__(self, state_dim, action_dim, config):
        super(ForwardModelNetwork, self).__init__(state_dim, action_dim, config)
        self._action = nn.Linear(action_dim, state_dim)
        self._hidden0 = nn.Linear(state_dim, config.forward_model_h1)
        self._hidden1 = nn.Linear(config.forward_model_h1 + state_dim, config.forward_model_h2)
        self._output = nn.Linear(config.forward_model_h2, state_dim)
        self.init()

    def forward(self, state, action):
        ax = torch.relu(self._action(action))
        x = torch.relu(self._hidden0(state))
        x = torch.cat([x, ax], state.ndim - 1)
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
        self._hidden0 = nn.Linear(state_dim + action_dim, config.metacritic_h1)
        self._hidden1 = nn.Linear(config.metacritic_h1, config.metacritic_h2)
        self._output = nn.Linear(config.metacritic_h2, 1)

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


def run_baseline(config):
    env = gym.make('TargetNavigate-v0')
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentNoisyDDPG('TargetNavigate-v0', env, config)

    for i in range(config.trials):
        actor = Actor(state_dim, action_dim, config)
        critic = Critic(state_dim, action_dim, config)
        memory = ExperienceReplayBuffer(config.memory_size)
        agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)
        experiment.run_baseline(agent, i)

    env.close()


def run_forward_model(config):
    env = gym.make('TargetNavigate-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentNoisyDDPG('TargetNavigate-v0', env, config)

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
    env = gym.make('TargetNavigate-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentNoisyDDPG('TargetNavigate-v0', env, config)

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
            metacritic = MetaLearnerMotivation(MetaLearnerNetwork(state_dim, action_dim, config), forward_model, config.metacritic_lr,
                                               config.metacritic_variant, env._max_episode_steps * 10, config.metacritic_eta,
                                               memory, config.metacritic_batch_size)
        else:
            metacritic = MetaLearnerMotivation(MetaLearnerNetwork(state_dim, action_dim, config), forward_model, config.metacritic_lr,
                                               config.metacritic_variant, env._max_episode_steps * 10, config.metacritic_eta)

        agent.add_motivation_module(metacritic)

        experiment.run_metalearner_model(agent, i)

    env.close()