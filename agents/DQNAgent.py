import torch

from algorithms.DQN import DQN
from algorithms.ReplayBuffer import ExperienceReplayBuffer
from modules.DQN_Modules import DQNSimpleNetwork


class DQNAgent:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network = None
        self.algorithm = None
        self.memory = None
        self.config = config

    def get_action(self, state):
        return self.network.value(state)

    def convert_action(self, action):
        a = torch.argmax(action, dim=1).numpy()

        if len(a) == 1:
            a = a.item()

        return a

    def save(self, path):
        torch.save(self.network.state_dict(), path + '.pth')

    def load(self, path):
        self.network.load_state_dict(torch.load(path + '.pth'))


class DQNSimpleAgent(DQNAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DQNSimpleNetwork(state_dim, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.algorithm = DQN(self.network, config.critic_lr, config.gamma)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        indices = self.memory.indices(self.config.batch_size)
        self.algorithm.train_sample(self.memory, indices)
