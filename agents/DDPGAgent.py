import torch

from algorithms.DDPG2 import DDPG2
from algorithms.ReplayBuffer import ExperienceReplayBuffer
from modules.DDPG_Modules import DDPGSimpleNetwork, DDPGAerisNetwork, DDPGAerisNetworkFM, DDPGAerisNetworkFME, DDPGAerisNetworkIM, DDPGAerisNetworkM2
from motivation.ForwardModelMotivation import ForwardModelMotivation
from motivation.M2Motivation import M2Motivation


class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network = None
        self.algorithm = None

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


class DDPGSimpleAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim)
        self.network = DDPGSimpleNetwork(state_dim, action_dim, config)
        memory = ExperienceReplayBuffer(config.memory_size)
        self.algorithm = DDPG2(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)


class DDPGAerisAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim)
        self.network = DDPGAerisNetwork(state_dim, action_dim, config)
        memory = ExperienceReplayBuffer(config.memory_size)
        self.algorithm = DDPG2(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)


class DDPGAerisForwardModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim)
        self.network = DDPGAerisNetworkFM(state_dim, action_dim, config)
        memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = ForwardModelMotivation(self.network.forward_model, config.forward_model_lr, config.forward_model_eta, config.forward_model_variant, memory, config.forward_model_batch_size, config.device)
        self.algorithm = DDPG2(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size, self.motivation)

    def train(self, state0, action0, state1, reward, mask):
        self.algorithm.train(state0, action0, state1, reward, mask)
        self.motivation.train(state0, action0, state1)


class DDPGAerisForwardModelEncoderAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim)
        self.network = DDPGAerisNetworkFME(state_dim, action_dim, config)
        memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = ForwardModelMotivation(self.network.forward_model, config.forward_model_lr, config.forward_model_eta, config.forward_model_variant, memory, config.forward_model_batch_size, config.device)
        self.algorithm = DDPG2(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size, self.motivation)

    def train(self, state0, action0, state1, reward, mask):
        self.algorithm.train(state0, action0, state1, reward, mask)
        self.motivation.train(state0, action0, state1)


class DDPGAerisInverseModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim)
        self.network = DDPGAerisNetworkIM(state_dim, action_dim, config)
        memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = ForwardModelMotivation(self.network.inverse_model, config.forward_model_lr, config.forward_model_eta, config.forward_model_variant, memory, config.forward_model_batch_size, config.device)
        self.algorithm = DDPG2(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size, self.motivation)

    def train(self, state0, action0, state1, reward, mask):
        self.algorithm.train(state0, action0, state1, reward, mask)
        self.motivation.train(state0, action0, state1)


class DDPGAerisM2ModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim)
        self.network = DDPGAerisNetworkM2(state_dim, action_dim, config)
        memory = ExperienceReplayBuffer(config.memory_size)
        gate_memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = M2Motivation(self.network, config.forward_model_lr, config.gamma, config.tau, config.forward_model_eta, memory, gate_memory, config.forward_model_batch_size)
        self.algorithm = DDPG2(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size, self.motivation)

    def train(self, state0, im0, action0, weight, state1, im1, reward, mask):
        self.algorithm.train(state0, action0, state1, reward, mask)
        self.motivation.train(state0, im0, action0, weight, state1, im1, reward, mask)
