import torch

from algorithms.DDPG2 import DDPG2
from algorithms.ReplayBuffer import ExperienceReplayBuffer
from modules.DDPG_Modules import DDPGNetwork


class DDPGAgent:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network = DDPGNetwork(state_dim, action_dim, config)

        memory = ExperienceReplayBuffer(config.memory_size)
        self.algorithm = DDPG2(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)

    def get_action(self, state):
        action = self.network.action(state)
        return action.detach()

    def convert_action(self, action):
        return action.squeeze(0).numpy()

    def train(self, state0, action0, state1, reward, mask):
        self.algorithm.train(state0, action0, state1, reward, mask)

    def save(self, path):
        torch.save(self.network.state_dict(), path + '.pth')

    def load(self, path):
        self.network.load_state_dict(torch.load(path + '.pth'))