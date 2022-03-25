import torch

from agents.PPOAgent import PPOAgent
from algorithms.PPO import PPO, MODE
from algorithms.ReplayBuffer import GenericTrajectoryBuffer, GenericAsyncTrajectoryBuffer
from modules.dop_models.DOPModelAtari import DOPControllerAtari
from modules.PPO_AtariModules import PPOAtariNetworkFM, PPOAtariNetwork, PPOAtariNetworkRND, PPOAtariNetworkQRND, PPOAtariNetworkDOP, PPOAtariMotivationNetwork, PPOAtariNetworkSRRND, \
    PPOAtariNetworkDOP, PPOAtariNetworkCND
from motivation.DOPMotivation import DOPMotivation
from motivation.Encoder import Encoder, DDMEncoder
from motivation.ForwardModelMotivation import ForwardModelMotivation
from motivation.RNDMotivation import RNDMotivation, QRNDMotivation


class PPOAtariAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, action_type, config)
        self.network = PPOAtariNetwork(input_shape, action_dim, config, head=action_type).to(config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=False)


class PPOAtariRNDAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, action_type, config)
        self.network = PPOAtariNetworkRND(input_shape, action_dim, config, head=action_type).to(config.device)
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


class PPOAtariSRRNDAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, action_type, config)
        self.network = PPOAtariNetworkSRRND(input_shape, 512, action_dim, config, head=action_type).to(config.device)
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


class PPOAtariCNDAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, action_type, config)
        self.network = PPOAtariNetworkCND(input_shape, 512, action_dim, config, head=action_type).to(config.device)
        self.encoder = Encoder(self.network.encoder, 0.00001, config.device)
        self.encoder_memory = GenericTrajectoryBuffer(config.trajectory_size // 8, config.batch_size, config.n_env)
        self.motivation_memory = GenericTrajectoryBuffer(config.trajectory_size, config.batch_size, config.n_env)
        self.motivation = RNDMotivation(self.network.cnd_model, config.motivation_lr, config.motivation_eta, config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ext_adv_scale=2, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=True)

    def train(self, state0, features0, value, action0, probs0, state1, features1, reward, mask):
        self.memory.add(state=features0.cpu(), value=value.cpu(), action=action0.cpu(), prob=probs0.cpu(), reward=reward.cpu(), mask=mask.cpu())
        self.encoder_memory.add(state=state0.cpu(), next_state=state1.cpu())
        self.motivation_memory.add(state=state0.cpu())

        indices = self.memory.indices()
        encoder_indices = self.encoder_memory.indices()
        motivation_indices = self.motivation_memory.indices()

        self.algorithm.train(self.memory, indices)
        self.motivation.train(self.motivation_memory, motivation_indices)
        self.encoder.train(self.encoder_memory, encoder_indices)

        if indices is not None:
            self.memory.clear()
        if encoder_indices is not None:
            self.encoder_memory.clear()
        if motivation_indices is not None:
            self.motivation_memory.clear()

    def get_features(self, state):
        features = self.network.encoder(state).detach()

        return features


class PPOAtariQRNDAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, action_type, config)
        self.network = PPOAtariNetworkQRND(input_shape, 512, action_dim, config, head=action_type).to(config.device)
        self.encoder = Encoder(self.network.encoder, 0.00001, config.device)
        self.encoder_memory = GenericTrajectoryBuffer(config.trajectory_size // 8, config.batch_size, config.n_env)
        self.motivation_memory = GenericTrajectoryBuffer(config.trajectory_size, config.batch_size, config.n_env)
        self.motivation = QRNDMotivation(self.network.qrnd_model, config.motivation_lr, config.motivation_eta, config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ext_adv_scale=2, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=True)

    def train(self, state0, features0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state=features0.cpu(), value=value.cpu(), action=action0.cpu(), prob=probs0.cpu(), reward=reward.cpu(), mask=mask.cpu())
        self.encoder_memory.add(state=state0.cpu(), next_state=state1.cpu())
        self.motivation_memory.add(state=state0.cpu(), action=action0.cpu())

        indices = self.memory.indices()
        encoder_indices = self.encoder_memory.indices()
        motivation_indices = self.motivation_memory.indices()

        self.algorithm.train(self.memory, indices)
        self.motivation.train(self.motivation_memory, motivation_indices)
        self.encoder.train(self.encoder_memory, encoder_indices)

        if indices is not None:
            self.memory.clear()
        if encoder_indices is not None:
            self.encoder_memory.clear()
        if motivation_indices is not None:
            self.motivation_memory.clear()

    def get_features(self, state):
        features = self.network.encoder(state).detach()

        return features


class PPOAtariDOPControllerAgent(PPOAgent):
    def __init__(self, network, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, action_type, config)
        self.network = network.to(config.device)
        self.memory = GenericAsyncTrajectoryBuffer(512, 512, config.n_env)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, 512, 512,
                             config.beta, config.gamma, ext_adv_scale=1, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=1,
                             device=config.device, motivation=False)

    def train(self, state0, value, action0, probs0, reward, mask):
        self.memory.add(self.network.aggregator_indices(), state=state0.cpu(), value=value.cpu(), action=action0.cpu(), prob=probs0.cpu(), reward=reward.cpu(), mask=mask.cpu())
        indices = self.memory.indices()
        self.algorithm.train(self.memory, indices)
        if indices is not None:
            self.memory.clear()


class PPOAtariDOPAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, action_type, config)

        gamma = config.gamma.split(',')
        self.head_count = config.dop_heads
        self.channels = input_shape[0]
        self.h = input_shape[1]
        self.w = input_shape[2]

        self.memory_external = GenericTrajectoryBuffer(config.trajectory_size, config.batch_size, config.n_env)
        self.memory_internal = GenericTrajectoryBuffer(config.trajectory_size, config.batch_size // config.dop_heads, config.n_env)
        self.qrnd_memory = GenericTrajectoryBuffer(config.trajectory_size, config.batch_size, config.n_env)
        self.encoder_memory = GenericTrajectoryBuffer(config.trajectory_size // 8, config.batch_size, config.n_env)

        self.network = PPOAtariNetworkDOP(input_shape, action_dim, config, action_type).to(config.device)
        self.encoder = Encoder(self.network.encoder, 1e-5, config.device)
        self.motivation = QRNDMotivation(self.network.qrnd_model, config.motivation_lr, config.motivation_eta, config.device)
        self.algorithm_external = PPO(self.network.dop_actor, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                                      config.beta, gamma[0], ext_adv_scale=1, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                                      device=config.device, motivation=False, mode=MODE.gate)

        self.algorithm_internal = PPO(self.network.dop_actor, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size // config.dop_heads, config.trajectory_size,
                                      config.beta, gamma[1], ext_adv_scale=1, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                                      device=config.device, motivation=False, mode=MODE.multicritic)

        self.controller = PPOAtariDOPControllerAgent(self.network.dop_controller, input_shape, action_dim, config, action_type)

    def train(self, features0_0, features0_1, state0, value, action0, probs0, head_value, head_action, head_probs, state1, reward0_0, reward0_1, mask0_0, mask0_1):
        self.controller.train(features0_1, head_value, head_action, head_probs, reward0_1, mask0_1)

        index = head_action.argmax(dim=1, keepdim=True).unsqueeze(-1)
        gated_action = torch.gather(action0, dim=1, index=index.repeat(1, 1, action0.shape[2])).squeeze(1)
        gated_probs = torch.gather(probs0, dim=1, index=index.repeat(1, 1, probs0.shape[2])).squeeze(1)
        gated_value = torch.gather(value[:, :, 0], dim=1, index=index.squeeze(-1))
        gated_reward = torch.gather(reward0_0[:, :, 0], dim=1, index=index.squeeze(-1))

        # value_mask = head_action.unsqueeze(-1).repeat(1, 1, 2) * 2 - 1
        # value_mask[:, :, 1] = 1
        # gated_action = gated_action.unsqueeze(1).repeat(1, self.head_count, 1)

        self.memory_external.add(state=features0_0.cpu(), value=gated_value.cpu(), action=gated_action.cpu(), prob=gated_probs.cpu(), reward=gated_reward.cpu(), mask=mask0_0.cpu(),
                                 heads=head_action.cpu())
        indices = self.memory_external.indices()
        self.algorithm_external.train(self.memory_external, indices)
        if indices is not None:
            self.memory_external.clear()

        value = value[:, :, 1].unsqueeze(-1)
        reward0_0 = reward0_0[:, :, 1].unsqueeze(-1)

        self.memory_internal.add(state=features0_0.cpu(), value=value.cpu(), action=action0.cpu(), prob=probs0.cpu(), reward=reward0_0.cpu(),
                                 mask=mask0_0.unsqueeze(1).repeat(1, self.head_count, 1).cpu())
        indices = self.memory_internal.indices()
        self.algorithm_internal.train(self.memory_internal, indices)
        if indices is not None:
            self.memory_internal.clear()

        self.qrnd_memory.add(state=state0.cpu(), action=gated_action.cpu())
        indices = self.qrnd_memory.indices()
        self.motivation.train(self.qrnd_memory, indices)
        if indices is not None:
            self.qrnd_memory.clear()

        self.encoder_memory.add(state=state0.cpu(), next_state=state1.cpu())
        indices = self.encoder_memory.indices()
        self.encoder.train(self.encoder_memory, indices)
        if indices is not None:
            self.encoder_memory.clear()

    def get_action(self, features0_0, features0_1):
        value, action, probs, head_value, head_action, head_probs = self.network(features0_0, features0_1)
        selected_action, _ = self.network.dop_actor.select_action(head_action, action, probs)

        return value.detach(), action, probs.detach(), head_value.detach(), head_action, head_probs.detach(), selected_action.detach()

    def get_features(self, state):
        features0_0 = self.network.encoder(state).detach()
        features0_1 = self.network.dop_controller.state(features0_0)

        return features0_0, features0_1

    def extend_state(self, state):
        return state.unsqueeze(1).repeat(1, self.head_count, 1, 1, 1).view(-1, self.channels, self.h, self.w)


class PPOAtariForwardModelAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, action_type, config)
        self.network = PPOAtariNetworkFM(input_shape, action_dim, config, head=action_type).to(config.device)
        self.motivation = ForwardModelMotivation(self.network.forward_model, config.forward_model_lr, config.forward_model_eta, config.forward_model_variant, self.memory, config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ext_adv_scale=2, int_adv_scale=1, ppo_epochs=config.ppo_epochs, n_env=config.n_env,
                             device=config.device, motivation=True)

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state=state0.cpu(), value=value.cpu(), action=action0.cpu(), prob=probs0.cpu(), next_state=state1.cpu(), reward=reward.cpu(), mask=mask.cpu())
        indices = self.memory.indices()
        self.algorithm.train(self.memory, indices)
        self.motivation.train(indices)
        if indices is not None:
            self.memory.clear()
