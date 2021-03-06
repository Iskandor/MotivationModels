import gym
import gym_aeris.envs
import torch
from torch import nn

from algorithms.DDPG import DDPGCritic, DDPGActor, DDPG
from algorithms.ReplayBuffer import ExperienceReplayBuffer
from experiment.ddpg_experiment import ExperimentDDPG
from modules import ARCH
from modules.forward_models.ForwardModel import ForwardModel
from modules.forward_models.RND_ForwardModel import RND_ForwardModel
from motivation.ForwardModelMotivation import ForwardModelMotivation
from motivation.MateCriticMotivation import MetaCriticMotivation


class Critic(DDPGCritic):
    def __init__(self, input_shape, action_dim, config):
        super(Critic, self).__init__(input_shape, action_dim)

        self.channels = input_shape[0]
        self.width = input_shape[1]

        fc_count = config.critic_kernels_count * self.width // 4

        self.layers = [
            nn.Conv1d(self.channels + action_dim, config.critic_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_count, config.critic_h1),
            nn.ReLU(),
            nn.Linear(config.critic_h1, 1)]

        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.xavier_uniform_(self.layers[3].weight)
        nn.init.uniform_(self.layers[5].weight, -0.003, 0.003)

        self._network = nn.Sequential(*self.layers)

    def forward(self, state, action):
        a = action.unsqueeze(state.ndim - 1).repeat(1, 1, state.shape[2])
        x = torch.cat([state, a], dim=1)
        value = self._network(x)
        return value


class Actor(DDPGActor):
    def __init__(self, input_shape, action_dim, config):
        super(Actor, self).__init__(input_shape, action_dim)

        self.channels = input_shape[0]
        self.width = input_shape[1]

        fc_count = config.actor_kernels_count * self.width // 4

        self.layers = [
            nn.Conv1d(self.channels, config.actor_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_count, config.actor_h1),
            nn.ReLU(),
            nn.Linear(config.actor_h1, action_dim),
            nn.Tanh()]

        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.xavier_uniform_(self.layers[3].weight)
        nn.init.uniform_(self.layers[5].weight, -0.3, 0.3)

        self._network = nn.Sequential(*self.layers)

    def forward(self, state):
        if state.ndim == 2:
            state = state.unsqueeze(0)
            policy = self._network(state).squeeze(0)
        else:
            policy = self._network(state)
        return policy


class MetaCritic(nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(MetaCritic, self).__init__()

        self.channels = input_shape[0]

        self.layers = [
            nn.Conv1d(self.channels + action_dim, config.metacritic_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(config.metacritic_kernels_count, config.metacritic_kernels_count, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(config.metacritic_kernels_count, config.metacritic_kernels_count, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(config.metacritic_kernels_count, config.metacritic_kernels_count, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(config.metacritic_kernels_count, config.metacritic_kernels_count, kernel_size=8, stride=4, padding=2, output_padding=0),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(in_features=64 * config.metacritic_kernels_count, out_features=1, bias=True)
        ]

        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.xavier_uniform_(self.layers[2].weight)
        nn.init.xavier_uniform_(self.layers[4].weight)
        nn.init.xavier_uniform_(self.layers[6].weight)
        nn.init.xavier_uniform_(self.layers[8].weight)
        nn.init.uniform_(self.layers[11].weight, -0.3, 0.3)

        self._network = nn.Sequential(*self.layers)

    def forward(self, state, action):
        if state.ndim == 3:
            a = action.unsqueeze(2).repeat(1, 1, state.shape[2])
            x = torch.cat([state, a], dim=1)
        if state.ndim == 2:
            a = action.unsqueeze(1).repeat(1, state.shape[1])
            x = torch.cat([state, a], dim=0).unsqueeze(0)

        value = self._network(x).squeeze(0)
        return value


def run_baseline(config, i):
    env = gym_aeris.envs.TargetNavigateEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('TargetNavigate-v0', env, config)

    actor = Actor(state_dim, action_dim, config)
    critic = Critic(state_dim, action_dim, config)
    memory = ExperienceReplayBuffer(config.memory_size)
    agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)
    experiment.run_baseline(agent, i)

    env.close()


def run_forward_model(config, i):
    env = gym_aeris.envs.TargetNavigateEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('TargetNavigate-v0', env, config)

    actor = Actor(state_dim, action_dim, config)
    critic = Critic(state_dim, action_dim, config)
    memory = ExperienceReplayBuffer(config.memory_size)

    agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)

    if hasattr(config, 'forward_model_batch_size'):
        forward_model = ForwardModelMotivation(ForwardModel(state_dim, action_dim, config, ARCH.aeris), config.forward_model_lr, config.forward_model_eta,
                                               config.forward_model_variant, 1000 * 10,
                                               memory, config.forward_model_batch_size)
    else:
        forward_model = ForwardModelMotivation(ForwardModel(state_dim, action_dim, config, ARCH.aeris), config.forward_model_lr, config.forward_model_eta,
                                               config.forward_model_variant, 1000 * 10)

    agent.add_motivation_module(forward_model)

    experiment.run_forward_model(agent, i)

    env.close()


def run_rnd_forward_model(config, i):
    env = gym_aeris.envs.TargetNavigateEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('TargetNavigate-v0', env, config)

    actor = Actor(state_dim, action_dim, config)
    critic = Critic(state_dim, action_dim, config)
    memory = ExperienceReplayBuffer(config.memory_size)

    agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)

    if hasattr(config, 'forward_model_batch_size'):
        forward_model = ForwardModelMotivation(RND_ForwardModel(state_dim, action_dim, config, ARCH.aeris), config.forward_model_lr, config.forward_model_eta,
                                               config.forward_model_variant, 1000 * 10,
                                               memory, config.forward_model_batch_size)
    else:
        forward_model = ForwardModelMotivation(RND_ForwardModel(state_dim, action_dim, config, ARCH.aeris), config.forward_model_lr, config.forward_model_eta,
                                               config.forward_model_variant, 1000 * 10)

    agent.add_motivation_module(forward_model)

    experiment.run_forward_model(agent, i)

    env.close()


def run_metalearner_model(config, i):
    env = gym_aeris.envs.TargetNavigateEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('TargetNavigate-v0', env, config)

    actor = Actor(state_dim, action_dim, config)
    critic = Critic(state_dim, action_dim, config)
    memory = ExperienceReplayBuffer(config.memory_size)

    agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)

    if hasattr(config, 'forward_model_batch_size'):
        forward_model = ForwardModelMotivation(ForwardModel(state_dim, action_dim, config, ARCH.aeris), config.forward_model_lr, config.forward_model_eta,
                                               config.forward_model_variant, 1000 * 10,
                                               memory, config.forward_model_batch_size)
    else:
        forward_model = ForwardModelMotivation(ForwardModel(state_dim, action_dim, config, ARCH.aeris), config.forward_model_lr, config.forward_model_eta,
                                               config.forward_model_variant, 1000 * 10)

    if hasattr(config, 'metacritic_batch_size'):
        metacritic = MetaCriticMotivation(MetaCritic(state_dim, action_dim, config), forward_model, config.metacritic_lr, state_dim,
                                          config.metacritic_variant, 1000 * 10, config.metacritic_eta,
                                          memory, config.metacritic_batch_size)
    else:
        metacritic = MetaCriticMotivation(MetaCritic(state_dim, action_dim, config), forward_model, config.metacritic_lr, state_dim,
                                          config.metacritic_variant, 1000 * 10, config.metacritic_eta)

    agent.add_motivation_module(metacritic)

    experiment.run_metalearner_model(agent, i)

    env.close()
