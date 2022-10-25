import torch

from agents.PPOAgent import PPOAgent
from algorithms.PPO import PPO, MODE
from algorithms.ReplayBuffer import GenericTrajectoryBuffer, GenericAsyncTrajectoryBuffer
from modules.PPO_ProcgenModules import PPOProcgenNetwork, PPOProcgenNetworkFEDRef, PPOProcgenNetworkRND, PPOProcgenNetworkFWD, PPOProcgenNetworkICM, PPOProcgenNetworkSRRND, PPOProcgenNetworkCND, PPOProcgenNetworkQRND
from motivation.CNDMotivation import CNDMotivation
from motivation.DOPMotivation import DOPMotivation
from motivation.Encoder import Encoder
from motivation.FEDRefMotivation import FEDRefMotivation
from motivation.ForwardModelMotivation import ForwardModelMotivation
from motivation.RNDMotivation import RNDMotivation, QRNDMotivation


class PPOProcgenAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, action_type, config)
        self.network = PPOProcgenNetwork(input_shape, action_dim, config, head=action_type).to(config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=False)


class PPOProcgenFEDRefAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, action_type, config)
        self.network = PPOProcgenNetworkFEDRef(input_shape, action_dim, config, head=action_type).to(config.device)
        self.motivation_memory = GenericTrajectoryBuffer(config.trajectory_size, config.batch_size, config.n_env)
        self.motivation = FEDRefMotivation(self.network.fed_ref_model, config.motivation_lr, config.motivation_eta, config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ext_adv_scale=2, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=True)

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state=state0.cpu(), value=value.cpu(), action=action0.cpu(), prob=probs0.cpu(), reward=reward.cpu(), mask=mask.cpu())
        self.motivation_memory.add(state=state0.cpu())

        indices = self.memory.indices()
        motivation_indices = self.motivation_memory.indices()

        if indices is not None:
            self.algorithm.train(self.memory, indices)
            self.memory.clear()

        if motivation_indices is not None:
            self.motivation.train(self.motivation_memory, motivation_indices)
            self.motivation_memory.clear()

    def get_action(self, state):
        value, action, probs = self.network(state)
        features = self.network.features(state)

        return features.detach(), value.detach(), action, probs.detach()


class PPOProcgenRNDAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, action_type, config)
        self.network = PPOProcgenNetworkRND(input_shape, action_dim, config, head=action_type).to(config.device)
        self.motivation = RNDMotivation(self.network.rnd_model, config.motivation_lr, config.motivation_eta, config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ext_adv_scale=2, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=True)

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state=state0.cpu(), value=value.cpu(), action=action0.cpu(), prob=probs0.cpu(), reward=reward.cpu(), mask=mask.cpu())
        indices = self.memory.indices()
        self.algorithm.train(self.memory, indices)
        self.motivation.train(self.memory, indices)
        if indices is not None:
            self.memory.clear()


class PPOProcgenFWDAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, action_type, config)
        self.network = PPOProcgenNetworkFWD(input_shape, action_dim, config, head=action_type).to(config.device)
        self.motivation = ForwardModelMotivation(self.network.forward_model, config.motivation_lr, config.motivation_eta, config.forward_model_variant, config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ext_adv_scale=2, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=True)

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state=state0.cpu(), value=value.cpu(), action=action0.cpu(), prob=probs0.cpu(), next_state=state1.cpu(), reward=reward.cpu(), mask=mask.cpu())
        indices = self.memory.indices()
        self.algorithm.train(self.memory, indices)
        self.motivation.train(self.memory, indices)
        if indices is not None:
            self.memory.clear()


class PPOProcgenICMAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, action_type, config)
        self.network = PPOProcgenNetworkICM(input_shape, action_dim, config, head=action_type).to(config.device)
        self.motivation = ForwardModelMotivation(self.network.forward_model, config.motivation_lr, config.motivation_eta, config.forward_model_variant, config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ext_adv_scale=2, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=True)

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state=state0.cpu(), value=value.cpu(), action=action0.cpu(), prob=probs0.cpu(), next_state=state1.cpu(), reward=reward.cpu(), mask=mask.cpu())
        indices = self.memory.indices()

        self.algorithm.train(self.memory, indices)
        self.motivation.train(self.memory, indices)
        if indices is not None:
            self.memory.clear()

    def get_action(self, state):
        value, action, probs = self.network(state)
        features = self.network.forward_model.encoder(state)

        return features.detach(), value.detach(), action, probs.detach()


class PPOProcgenSRRNDAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, action_type, config)
        self.network = PPOProcgenNetworkSRRND(input_shape, 512, action_dim, config, head=action_type).to(config.device)
        self.encoder = Encoder(self.network.encoder, 0.00001, config.device)
        self.encoder_memory = GenericTrajectoryBuffer(config.trajectory_size, config.batch_size, config.n_env)
        self.motivation = RNDMotivation(self.network.rnd_model, config.motivation_lr, config.motivation_eta, config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ext_adv_scale=2, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=True)

    def train(self, state0, features0, value, action0, probs0, state1, features1, reward, mask):
        self.memory.add(state=features0.cpu(), value=value.cpu(), action=action0.cpu(), prob=probs0.cpu(), reward=reward.cpu(), mask=mask.cpu())
        self.encoder_memory.add(state=state0.cpu(), next_state=state1.cpu())
        indices = self.memory.indices()
        memory_indices = self.encoder_memory.indices()
        self.algorithm.train(self.memory, indices)
        self.motivation.train(self.encoder_memory, memory_indices)
        self.encoder.train(self.encoder_memory, memory_indices)
        if indices is not None:
            self.memory.clear()
        if memory_indices is not None:
            self.encoder_memory.clear()

    def get_features(self, state):
        features = self.network.encoder(state).detach()

        return features


class PPOProcgenCNDAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, action_type, config)
        self.network = PPOProcgenNetworkCND(input_shape, action_dim, config, head=action_type).to(config.device)
        self.motivation_memory = GenericTrajectoryBuffer(config.trajectory_size, config.batch_size, config.n_env)
        self.motivation = CNDMotivation(self.network.cnd_model, config.motivation_lr, config.motivation_eta, config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ext_adv_scale=2, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=True)

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state=state0.cpu(), value=value.cpu(), action=action0.cpu(), prob=probs0.cpu(), reward=reward.cpu(), mask=mask.cpu())
        self.motivation_memory.add(state=state0.cpu(), next_state=state1.cpu())

        indices = self.memory.indices()
        motivation_indices = self.motivation_memory.indices()

        if indices is not None:
            self.algorithm.train(self.memory, indices)
            self.memory.clear()

        if motivation_indices is not None:
            self.motivation.train(self.motivation_memory, motivation_indices)
            self.motivation_memory.clear()

    def get_action(self, state):
        value, action, probs = self.network(state)
        features = self.network.cnd_model.target_model(self.network.cnd_model.preprocess(state))

        return features.detach(), value.detach(), action, probs.detach()


class PPOProcgenQRNDAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, action_type, config)
        self.network = PPOProcgenNetworkQRND(input_shape, action_dim, config, head=action_type).to(config.device)
        self.motivation_memory = GenericTrajectoryBuffer(config.trajectory_size, config.batch_size, config.n_env)
        self.motivation = QRNDMotivation(self.network.qrnd_model, config.motivation_lr, config.motivation_eta, config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ext_adv_scale=2, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=True)

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state=state0.cpu(), value=value.cpu(), action=action0.cpu(), prob=probs0.cpu(), reward=reward.cpu(), mask=mask.cpu())
        self.motivation_memory.add(state=state0.cpu(), action=action0.cpu())

        indices = self.memory.indices()
        motivation_indices = self.motivation_memory.indices()

        if indices is not None:
            self.algorithm.train(self.memory, indices)
            self.memory.clear()
        if motivation_indices is not None:
            self.motivation.train(self.motivation_memory, motivation_indices)
            self.motivation_memory.clear()

    def get_features(self, state):
        features = self.network.encoder(state).detach()

        return features
