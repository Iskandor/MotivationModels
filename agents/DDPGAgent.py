import torch

from algorithms.DDPG2 import DDPG2
from algorithms.ReplayBuffer import ExperienceReplayBuffer, M2ReplayBuffer
from modules.DDPG_Modules import DDPGSimpleNetwork, DDPGAerisNetwork, DDPGAerisNetworkFM, DDPGAerisNetworkFME, DDPGAerisNetworkIM, DDPGAerisNetworkM2, DDPGAerisNetworkFIM, DDPGAerisNetworkSU
from motivation.ForwardInverseModelMotivation import ForwardInverseModelMotivation
from motivation.ForwardModelMotivation import ForwardModelMotivation
from motivation.M2Motivation import M2Motivation
from motivation.MetaCriticMotivation import MetaCriticMotivation
from motivation.M2SMotivation import M2SMotivation


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
        return action.squeeze(0).numpy()

    def save(self, path):
        torch.save(self.network.state_dict(), path + '.pth')

    def load(self, path):
        self.network.load_state_dict(torch.load(path + '.pth'))


class DDPGSimpleAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGSimpleNetwork(state_dim, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.algorithm = DDPG2(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        indices = self.memory.indices(self.config.batch_size)
        self.algorithm.train_sample(indices)


class DDPGAerisAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetwork(state_dim, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.algorithm = DDPG2(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        indices = self.memory.indices(self.config.batch_size)
        self.algorithm.train_sample(indices)


class DDPGAerisForwardModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetworkFM(state_dim, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = ForwardModelMotivation(self.network.forward_model, config.forward_model_lr, config.forward_model_eta, config.forward_model_variant, self.memory, config.forward_model_batch_size,
                                                 config.device)
        self.algorithm = DDPG2(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size, self.motivation)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.forward_model_batch_size))


class DDPGAerisForwardModelEncoderAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetworkFME(state_dim, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = ForwardModelMotivation(self.network.forward_model, config.forward_model_lr, config.forward_model_eta, config.forward_model_variant, self.memory, config.forward_model_batch_size,
                                                 config.device)
        self.algorithm = DDPG2(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size, self.motivation)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.forward_model_batch_size))


class DDPGAerisInverseModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetworkIM(state_dim, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = ForwardModelMotivation(self.network.inverse_model, config.forward_model_lr, config.forward_model_eta, config.forward_model_variant, self.memory, config.forward_model_batch_size,
                                                 config.device)
        self.algorithm = DDPG2(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size, self.motivation)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.forward_model_batch_size))


class DDPGAerisForwardInverseModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetworkFIM(state_dim, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = ForwardInverseModelMotivation(self.network.forward_model, config.forward_model_lr, self.network.inverse_model, config.forward_model_lr,
                                                        0.5, config.forward_model_eta,
                                                        config.forward_model_variant, self.memory, config.forward_model_batch_size, config.device)
        self.algorithm = DDPG2(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size, self.motivation)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.forward_model_batch_size))


class DDPGAerisGatedMetacriticModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetworkSU(state_dim, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = MetaCriticMotivation(self.network, config.metacritic_lr, config.metacritic_variant, config.metacritic_eta, self.memory, config.metacritic_batch_size, config.device)
        self.algorithm = DDPG2(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size, self.motivation)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.forward_model_batch_size))


class DDPGAerisM2ModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetworkM2(state_dim, action_dim, config)
        self.memory = M2ReplayBuffer(config.memory_size)
        self.motivation = M2Motivation(self.network, config.forward_model_lr, config.gamma, config.tau, config.forward_model_eta, self.memory, config.forward_model_batch_size)
        self.algorithm = DDPG2(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size, self.motivation)

    def train(self, state0, action0, state1, im0, error0, weight, im1, error1, reward, mask):
        gate_state0 = self.compose_gate_state(im0, error0)
        gate_state1 = self.compose_gate_state(im1, error1)
        self.memory.add(state0, action0, state1, gate_state0, weight, gate_state1, reward, mask)
        self.algorithm.train_sample(self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.forward_model_batch_size))

    def compose_gate_state(self, im, error):
        return torch.cat([im, error], dim=1)


class DDPGAerisM2SModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetworkFM(state_dim, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = M2SMotivation(self.network, config.forward_model_lr, config.forward_model_eta, self.memory, config.forward_model_batch_size, config.steps * 1e6)
        self.algorithm = DDPG2(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size, self.motivation)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.forward_model_batch_size))
