import torch

from agents.DDPGAgent import DDPGAgent
from algorithms.DDPG import DDPG
from algorithms.ReplayBuffer import ExperienceReplayBuffer, M2ReplayBuffer, MDPTrajectoryBuffer
from modules.DDPG_AerisModules import DDPGAerisNetwork, DDPGAerisNetworkFM, DDPGAerisNetworkFME, DDPGAerisNetworkIM, DDPGAerisNetworkFIM, DDPGAerisNetworkSU, DDPGAerisNetworkM2, DDPGAerisNetworkRND, \
    DDPGAerisNetworkQRND, DDPGAerisNetworkDOP, DDPGAerisNetworkDOPV2, DDPGAerisNetworkDOPV2Q, DDPGAerisNetworkDOPRef, DDPGAerisNetworkSURND, DDPGAerisNetworkDOPV3, DDPGAerisNetworkVanillaDOP
from motivation.DOPMotivation import DOPMotivation, DOPV2QMotivation
from motivation.ForwardInverseModelMotivation import ForwardInverseModelMotivation
from motivation.ForwardModelMotivation import ForwardModelMotivation
from motivation.M2Motivation import M2Motivation
from motivation.M2SMotivation import M2SMotivation
from motivation.MetaCriticMotivation import MetaCriticMotivation, MetaCriticRNDMotivation
from motivation.RNDMotivation import RNDMotivation, QRNDMotivation


class DDPGAerisAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetwork(state_dim, action_dim, config).to(config.device)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, device=config.device)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        indices = self.memory.indices(self.config.batch_size)
        self.algorithm.train_sample(self.memory, indices)


class DDPGAerisForwardModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetworkFM(state_dim, action_dim, config).to(config.device)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = ForwardModelMotivation(self.network.forward_model, config.forward_model_lr, config.forward_model_eta, config.forward_model_variant, self.memory, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.motivation, device=config.device)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory, self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.forward_model_batch_size))


class DDPGAerisForwardModelEncoderAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetworkFME(state_dim, action_dim, config).to(config.device)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = ForwardModelMotivation(self.network.forward_model, config.forward_model_lr, config.forward_model_eta, config.forward_model_variant, self.memory, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.motivation, device=config.device)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory, self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.forward_model_batch_size))


class DDPGAerisInverseModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetworkIM(state_dim, action_dim, config).to(config.device)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = ForwardModelMotivation(self.network.inverse_model, config.forward_model_lr, config.forward_model_eta, config.forward_model_variant, self.memory, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.motivation, device=config.device)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory, self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.forward_model_batch_size))


class DDPGAerisForwardInverseModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetworkFIM(state_dim, action_dim, config).to(config.device)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = ForwardInverseModelMotivation(self.network.forward_model, config.forward_model_lr, self.network.inverse_model, config.forward_model_lr,
                                                        0.5, config.forward_model_eta,
                                                        config.forward_model_variant, self.memory, config.forward_model_batch_size, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.motivation, device=config.device)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory, self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.forward_model_batch_size))


class DDPGAerisGatedMetacriticModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetworkSU(state_dim, action_dim, config).to(config.device)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = MetaCriticMotivation(self.network, config.metacritic_lr, config.metacritic_variant, config.metacritic_eta, self.memory, config.metacritic_batch_size, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.motivation, device=config.device)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory, self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.forward_model_batch_size))


class DDPGAerisM2ModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetworkM2(state_dim, action_dim, config).to(config.device)
        self.memory = M2ReplayBuffer(config.memory_size)
        self.motivation = M2Motivation(self.network, config.forward_model_lr, config.gamma, config.tau, config.forward_model_eta, self.memory, config.forward_model_batch_size)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.motivation, device=config.device)

    def train(self, state0, action0, state1, im0, error0, weight, im1, error1, reward, mask):
        gate_state0 = self.compose_gate_state(im0, error0)
        gate_state1 = self.compose_gate_state(im1, error1)
        self.memory.add(state0, action0, state1, gate_state0, weight, gate_state1, reward, mask)
        self.algorithm.train_sample(self.memory, self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.forward_model_batch_size))

    def compose_gate_state(self, im, error):
        return torch.cat([im, error], dim=1)


class DDPGAerisM2SModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetworkFM(state_dim, action_dim, config).to(config.device)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = M2SMotivation(self.network, config.forward_model_lr, config.forward_model_eta, self.memory, config.forward_model_batch_size, config.steps * 1e6)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.motivation, device=config.device)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory, self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.forward_model_batch_size))


class DDPGAerisRNDModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetworkRND(state_dim, action_dim, config).to(config.device)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation_memory = MDPTrajectoryBuffer(self.config.forward_model_batch_size, self.config.forward_model_batch_size)
        self.motivation = RNDMotivation(self.network.rnd_model, config.forward_model_lr, config.motivation_eta, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.motivation, device=config.device)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.motivation_memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory, self.memory.indices(self.config.batch_size))
        indices = self.motivation_memory.indices()
        self.motivation.train(self.motivation_memory, indices)
        if indices is not None:
            self.motivation_memory.clear()


class DDPGAerisQRNDModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetworkQRND(state_dim, action_dim, config).to(config.device)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation_memory = MDPTrajectoryBuffer(self.config.forward_model_batch_size, self.config.forward_model_batch_size)
        self.motivation = QRNDMotivation(self.network.qrnd_model, config.forward_model_lr, config.motivation_eta, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.motivation, device=config.device)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.motivation_memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory, self.memory.indices(self.config.batch_size))
        indices = self.motivation_memory.indices()
        self.motivation.train(self.motivation_memory, indices)
        if indices is not None:
            self.motivation_memory.clear()


class DDPGAerisVanillaDOPAgent(DDPGAgent):
    def __init__(self, input_shape, action_dim, config):
        super().__init__(input_shape, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation_memory = MDPTrajectoryBuffer(self.config.forward_model_batch_size, self.config.forward_model_batch_size)
        self.network = DDPGAerisNetworkVanillaDOP(input_shape, action_dim, config).to(config.device)
        self.motivation = DOPMotivation(self.network.dop_model, config.forward_model_lr, config.actor_lr, config.motivation_eta, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, device=config.device, beta=config.beta)

    def get_actions(self, state):
        return self.network.all_actions(state).detach()

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.motivation_memory.add(state0, action0, state1, reward, mask)
        ddpg_indices = self.memory.indices(self.config.batch_size)
        self.algorithm.train_sample(self.memory, ddpg_indices)
        dop_indices = self.motivation_memory.indices()
        self.motivation.train(self.motivation_memory, dop_indices, self.memory, ddpg_indices)
        if dop_indices is not None:
            self.motivation_memory.clear()


class DDPGAerisDOPAgent(DDPGAgent):
    def __init__(self, input_shape, action_dim, config):
        super().__init__(input_shape, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation_memory = MDPTrajectoryBuffer(self.config.forward_model_batch_size, self.config.forward_model_batch_size)
        self.network = DDPGAerisNetworkDOP(input_shape, action_dim, config).to(config.device)
        self.motivation = DOPMotivation(self.network.dop_model, config.forward_model_lr, config.actor_lr, config.motivation_eta, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, device=config.device)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.motivation_memory.add(state0, action0, state1, reward, mask)
        ddpg_indices = self.memory.indices(self.config.batch_size)
        self.algorithm.train_sample(self.memory, ddpg_indices)
        dop_indices = self.motivation_memory.indices()
        self.motivation.train(self.motivation_memory, dop_indices, self.memory, ddpg_indices)
        if dop_indices is not None:
            self.motivation_memory.clear()

    def get_actions(self, state):
        return self.network.all_actions(state).detach()


class DDPGAerisDOPV2Agent(DDPGAgent):
    def __init__(self, input_shape, action_dim, config):
        super().__init__(input_shape, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation_memory = MDPTrajectoryBuffer(self.config.forward_model_batch_size, self.config.forward_model_batch_size)
        self.network = DDPGAerisNetworkDOPV2(input_shape, action_dim, config).to(config.device)
        self.motivation = DOPMotivation(self.network.dop_model, config.forward_model_lr, config.actor_lr, config.motivation_eta, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, device=config.device)

    def get_action(self, state):
        action = self.network.action(state)
        index = self.network.index()
        accuracy = self.network.accuracy()
        return action.detach(), index, accuracy

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.motivation_memory.add(state0, action0, state1, reward, mask)
        ddpg_indices = self.memory.indices(self.config.batch_size)
        self.algorithm.train_sample(self.memory, ddpg_indices)
        dop_indices = self.motivation_memory.indices()
        self.motivation.train(self.motivation_memory, dop_indices, self.memory, ddpg_indices)
        if dop_indices is not None:
            self.motivation_memory.clear()


class DDPGAerisDOPV2QAgent(DDPGAgent):
    def __init__(self, input_shape, action_dim, config):
        super().__init__(input_shape, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation_memory = MDPTrajectoryBuffer(self.config.forward_model_batch_size, self.config.forward_model_batch_size)
        self.network = DDPGAerisNetworkDOPV2Q(input_shape, action_dim, config).to(config.device)
        self.motivation = DOPV2QMotivation(self.network.dop_model, config.forward_model_lr, config.actor_lr, config.motivation_eta, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, device=config.device)

    def get_action(self, state):
        action = self.network.action(state)
        index = self.network.index()
        return action.detach(), index

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.motivation_memory.add(state0, action0, state1, reward, mask)
        ddpg_indices = self.memory.indices(self.config.batch_size)
        self.algorithm.train_sample(self.memory, ddpg_indices)
        dop_indices = self.motivation_memory.indices()
        self.motivation.train(self.motivation_memory, dop_indices, self.memory, ddpg_indices)
        if dop_indices is not None:
            self.motivation_memory.clear()


class DDPGAerisDOPV3Agent(DDPGAgent):
    def __init__(self, input_shape, action_dim, config):
        super().__init__(input_shape, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation_memory = MDPTrajectoryBuffer(self.config.forward_model_batch_size, self.config.forward_model_batch_size)
        self.network = DDPGAerisNetworkDOPV3(input_shape, action_dim, config).to(config.device)
        self.motivation = DOPMotivation(self.network.dop_model, config.forward_model_lr, config.actor_lr, config.motivation_eta, config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, motivation=self.motivation, device=config.device)

    def get_action(self, state):
        action = self.network.action(state)
        index = self.network.index()
        return action.detach(), index

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.motivation_memory.add(state0, action0, state1, reward, mask)
        ddpg_indices = self.memory.indices(self.config.batch_size)
        self.algorithm.train_sample(self.memory, ddpg_indices)
        dop_indices = self.motivation_memory.indices()
        self.motivation.train(self.motivation_memory, dop_indices, self.memory, ddpg_indices)
        if dop_indices is not None:
            self.motivation_memory.clear()


class DDPGAerisDOPRefAgent(DDPGAgent):
    def __init__(self, input_shape, action_dim, config):
        super().__init__(input_shape, action_dim, config)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.network = DDPGAerisNetworkDOPRef(input_shape, action_dim, config).to(config.device)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, device=config.device)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        ddpg_indices = self.memory.indices(self.config.batch_size)
        self.algorithm.train_sample(self.memory, ddpg_indices)


class DDPGAerisMetaCriticRNDModelAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.network = DDPGAerisNetworkSURND(state_dim, action_dim, config).to(config.device)
        self.memory = ExperienceReplayBuffer(config.memory_size)
        self.motivation = MetaCriticRNDMotivation(self.network.metacritic_model, config.metacritic_lr, config.metacritic_variant, config.metacritic_eta, self.memory, config.metacritic_batch_size)
        self.algorithm = DDPG(self.network, config.actor_lr, config.critic_lr, config.gamma, config.tau, self.motivation, device=config.device)

    def train(self, state0, action0, state1, reward, mask):
        self.memory.add(state0, action0, state1, reward, mask)
        self.algorithm.train_sample(self.memory, self.memory.indices(self.config.batch_size))
        self.motivation.train(self.memory.indices(self.config.metacritic_batch_size))
