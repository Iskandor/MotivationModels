import torch

from algorithms.DDPG import DDPG
from algorithms.ReplayBuffer import ExperienceReplayBuffer, M2ReplayBuffer, DOPReplayBuffer, PPOTrajectoryBuffer, MDPTrajectoryBuffer
from modules.DDPG_Modules import DDPGSimpleNetwork, DDPGAerisNetwork, DDPGAerisNetworkFM, DDPGAerisNetworkFME, DDPGAerisNetworkIM, DDPGAerisNetworkM2, DDPGAerisNetworkFIM, DDPGAerisNetworkSU, \
    DDPGAerisNetworkRND, DDPGAerisNetworkSURND, DDPGBulletNetwork, DDPGBulletNetworkFM, DDPGBulletNetworkSU, DDPGBulletNetworkRND, DDPGBulletNetworkSURND, DDPGBulletNetworkQRND, DDPGBulletNetworkDOP, \
    DDPGBulletNetworkDOPSimple, DDPGAerisNetworkQRND, DDPGAerisNetworkDOP, DDPGAerisNetworkDOPRef, DDPGAerisNetworkDOPV2
from motivation.ForwardInverseModelMotivation import ForwardInverseModelMotivation
from motivation.ForwardModelMotivation import ForwardModelMotivation
from motivation.M2Motivation import M2Motivation
from motivation.MetaCriticMotivation import MetaCriticMotivation, MetaCriticRNDMotivation
from motivation.M2SMotivation import M2SMotivation
from motivation.RNDMotivation import RNDMotivation, QRNDMotivation, DOPMotivation, DOPSimpleMotivation


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
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        indices = self.memory.indices(self.config.batch_size)
        self.algorithm.train_sample(indices)


class DDPGBulletAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGBulletNetwork(state_dim, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        indices = self.memory.indices(self.config.batch_size)
        self.algorithm.train_sample(indices)


class DDPGBulletForwardModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGBulletNetworkFM(state_dim, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = ForwardModelMotivation(self.network.forward_model, config.motivation_lr, config.motivation_eta, config.motivation_variant, self.memory, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size, self.motivation)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.motivation_batch_size))


class DDPGBulletGatedMetacriticModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGBulletNetworkSU(state_dim, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = MetaCriticMotivation(self.network, config.motivation_lr, config.motivation_variant, config.motivation_eta, self.memory, config.motivation_batch_size, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size, self.motivation)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.motivation_batch_size))


class DDPGBulletRNDModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGBulletNetworkRND(state_dim, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = RNDMotivation(self.network.rnd_model, config.motivation_lr, config.motivation_eta, self.memory, config.motivation_batch_size, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size, self.motivation)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.motivation_batch_size))


class DDPGBulletQRNDModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGBulletNetworkQRND(state_dim, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = QRNDMotivation(self.network.qrnd_model, config.motivation_lr, config.motivation_eta, self.memory, config.motivation_batch_size, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size, self.motivation)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.motivation_batch_size))


class DDPGBulletDOPSimpleModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGBulletNetworkDOPSimple(state_dim, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = DOPSimpleMotivation(self.network.dop_model, config.motivation_lr, config.motivation_eta, self.memory, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.motivation_batch_size))


class DDPGBulletDOPModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGBulletNetworkDOP(state_dim, action_dim, config)
        self.memory = DOPReplayBuffer(config.memory_size)
        self.motivation = DOPMotivation(self.network.dop_model, config.motivation_lr, config.motivation_eta, self.memory, config.motivation_batch_size, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size)

    def get_action(self, state):
        action = self.network.action(state)
        noise, index = self.network.noise(state, action)
        return action.detach(), noise.detach(), index.detach()

    def train(self, state0, action0, noise0, index, state1, reward, mask):
        self.memory.add(state0, action0, noise0, index, state1, reward, mask)
        self.algorithm.train_sample(self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.motivation_batch_size))


class DDPGBulletMetaCriticRNDModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGBulletNetworkSURND(state_dim, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = MetaCriticRNDMotivation(self.network.metacritic_model, config.motivation_lr, config.motivation_variant, config.motivation_eta, self.memory, config.motivation_batch_size,
                                                  config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size, self.motivation)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.motivation_batch_size))


class DDPGAerisAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetwork(state_dim, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        indices = self.memory.indices(self.config.batch_size)
        self.algorithm.train_sample(indices)


class DDPGAerisForwardModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetworkFM(state_dim, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = ForwardModelMotivation(self.network.forward_model, config.forward_model_lr, config.forward_model_eta, config.forward_model_variant, self.memory, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size, self.motivation)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.forward_model_batch_size))


class DDPGAerisForwardModelEncoderAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetworkFME(state_dim, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = ForwardModelMotivation(self.network.forward_model, config.forward_model_lr, config.forward_model_eta, config.forward_model_variant, self.memory, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size, self.motivation)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.forward_model_batch_size))


class DDPGAerisInverseModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetworkIM(state_dim, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = ForwardModelMotivation(self.network.inverse_model, config.forward_model_lr, config.forward_model_eta, config.forward_model_variant, self.memory, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size, self.motivation)

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
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size, self.motivation)

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
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size, self.motivation)

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
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size, self.motivation)

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
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size, self.motivation)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.forward_model_batch_size))


class DDPGAerisRNDModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetworkRND(state_dim, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation_memory = MDPTrajectoryBuffer(self.config.forward_model_batch_size, self.config.forward_model_batch_size)
        self.motivation = RNDMotivation(self.network.rnd_model, config.forward_model_lr, config.motivation_eta, self.motivation_memory)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size, self.motivation)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.motivation_memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory.indices(self.config.batch_size))
        indices = self.motivation_memory.indices()
        self.motivation.train(indices)
        if indices is not None:
            self.motivation_memory.clear()


class DDPGAerisQRNDModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetworkQRND(state_dim, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation_memory = MDPTrajectoryBuffer(self.config.forward_model_batch_size, self.config.forward_model_batch_size)
        self.motivation = QRNDMotivation(self.network.qrnd_model, config.forward_model_lr, config.motivation_eta, self.motivation_memory)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size, self.motivation)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.motivation_memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory.indices(self.config.batch_size))
        indices = self.motivation_memory.indices()
        self.motivation.train(indices)
        if indices is not None:
            self.motivation_memory.clear()


class DDPGAerisDOPAgent(DDPGAgent):
    def __init__(self, input_shape, action_dim, config):
        super().__init__(input_shape, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation_memory = MDPTrajectoryBuffer(self.config.forward_model_batch_size, self.config.forward_model_batch_size)
        self.network = DDPGAerisNetworkDOPV2(input_shape, action_dim, config).to(config.device)
        self.motivation = DOPMotivation(self.network.dop_model, config.forward_model_lr, config.actor_lr, config.motivation_eta, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size)

    def get_action(self, state):
        action = self.network.action(state)
        index = self.network.index()
        return action.detach(), index

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.motivation_memory.add(state0, action0, state1, reward, mask)
        ddpg_indices = self.memory.indices(self.config.batch_size)
        self.algorithm.train_sample(ddpg_indices)
        dop_indices = self.motivation_memory.indices()
        self.motivation.train(self.motivation_memory, dop_indices, self.memory, ddpg_indices)
        if dop_indices is not None:
            self.motivation_memory.clear()


class DDPGAerisDOPRefAgent(DDPGAgent):
    def __init__(self, input_shape, action_dim, config):
        super().__init__(input_shape, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.network = DDPGAerisNetworkDOPRef(input_shape, action_dim, config).to(config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        ddpg_indices = self.memory.indices(self.config.batch_size)
        self.algorithm.train_sample(ddpg_indices)


class DDPGAerisMetaCriticRNDModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetworkSURND(state_dim, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = MetaCriticRNDMotivation(self.network.metacritic_model, config.metacritic_lr, config.metacritic_variant, config.metacritic_eta, self.memory, config.metacritic_batch_size)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.memory, config.batch_size, self.motivation)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.metacritic_batch_size))
