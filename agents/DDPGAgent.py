import torch

from algorithms.DDPG import DDPG
from algorithms.ReplayBuffer import ExperienceReplayBuffer
from modules.DDPG_Modules import DDPGSimpleNetwork, DDPGBulletNetwork, DDPGBulletNetworkFM, DDPGBulletNetworkSU, DDPGBulletNetworkRND, DDPGBulletNetworkSURND, DDPGBulletNetworkQRND
from motivation.ForwardModelMotivation import ForwardModelMotivation
from motivation.MetaCriticMotivation import MetaCriticMotivation, MetaCriticRNDMotivation
from motivation.RNDMotivation import RNDMotivation, QRNDMotivation


class DDPGAgent:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network = None
        self.algorithm = None
        self.memory = None
        self.config = config

    def get_action(self, state):
        action = self.network.action(state)
        return action.detach()

    def convert_action(self, action):
        return action.squeeze(0).cpu().numpy()

    def save(self, path):
        torch.save(self.network.state_dict(), path + '.pth')

    def load(self, path):
        self.network.load_state_dict(torch.load(path + '.pth'))


class DDPGSimpleAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGSimpleNetwork(state_dim, action_dim, config).to(config.device)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, device=config.device)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        indices = self.memory.indices(self.config.batch_size)
        self.algorithm.train_sample(self.memory, indices)


class DDPGBulletAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGBulletNetwork(state_dim, action_dim, config).to(config.device)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, device=config.device)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        indices = self.memory.indices(self.config.batch_size)
        self.algorithm.train_sample(self.memory, indices)


class DDPGBulletForwardModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGBulletNetworkFM(state_dim, action_dim, config).to(config.device)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = ForwardModelMotivation(self.network.forward_model, config.motivation_lr, config.motivation_eta, config.motivation_variant, self.memory, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.motivation, device=config.device)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory, self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.motivation_batch_size))


class DDPGBulletGatedMetacriticModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGBulletNetworkSU(state_dim, action_dim, config).to(config.device)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = MetaCriticMotivation(self.network, config.motivation_lr, config.motivation_variant, config.motivation_eta, self.memory, config.motivation_batch_size, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.motivation, device=config.device)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory, self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.motivation_batch_size))


class DDPGBulletRNDModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGBulletNetworkRND(state_dim, action_dim, config).to(config.device)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = RNDMotivation(self.network.rnd_model, config.motivation_lr, config.motivation_eta, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.motivation, device=config.device)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory, self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory, self.memory.indices(self.config.motivation_batch_size))


class DDPGBulletQRNDModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGBulletNetworkQRND(state_dim, action_dim, config).to(config.device)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = QRNDMotivation(self.network.qrnd_model, config.motivation_lr, config.motivation_eta, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.motivation, device=config.device)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory, self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory, self.memory.indices(self.config.motivation_batch_size))


class DDPGBulletMetaCriticRNDModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGBulletNetworkSURND(state_dim, action_dim, config).to(config.device)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = MetaCriticRNDMotivation(self.network.metacritic_model, config.motivation_lr, config.motivation_variant, config.motivation_eta, self.memory, config.motivation_batch_size,
                                                  config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.motivation, device=config.device)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory, self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.motivation_batch_size))


