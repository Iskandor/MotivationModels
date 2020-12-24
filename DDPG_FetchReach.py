import gym
import torch

from algorithms.DDPG import DDPG, DDPGCritic, DDPGActor
from algorithms.ReplayBuffer import ExperienceReplayBuffer
from ddpg_noisy_experiment import ExperimentNoisyDDPG
from modules.NoisyLinear import NoisyLinear
from motivation.ForwardModelMotivation import ForwardModel


class Critic(DDPGCritic):
    def __init__(self, state_dim, action_dim, config):
        super(Critic, self).__init__(state_dim, action_dim)

        self._hidden0 = torch.nn.Linear(state_dim, config.critic_h1)
        self._hidden1 = torch.nn.Linear(config.critic_h1 + action_dim, config.critic_h2)
        self._output = torch.nn.Linear(config.critic_h2, 1)

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

        self._hidden0 = torch.nn.Linear(state_dim, config.actor_h1)
        self._hidden1 = torch.nn.Linear(config.actor_h1, config.actor_h2)
        self._output = NoisyLinear(config.actor_h2, action_dim)

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


def encode_state(state):
    achieved_goal = torch.tensor(state['achieved_goal'], dtype=torch.float32)
    desired_goal = torch.tensor(state['desired_goal'], dtype=torch.float32)
    return torch.cat((achieved_goal, desired_goal))


def run_baseline(config):
    env = gym.make('FetchReach-v1')
    state_dim = env.observation_space['achieved_goal'].shape[0] + env.observation_space['desired_goal'].shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentNoisyDDPG('FetchReach-v1', env, config)
    experiment.add_preprocess(encode_state)

    for i in range(config.trials):
        actor = Actor(state_dim, action_dim, config)
        critic = Critic(state_dim, action_dim, config)
        memory = ExperienceReplayBuffer(config.memory_size)
        agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)
        experiment.run_baseline(agent, i)

    env.close()
