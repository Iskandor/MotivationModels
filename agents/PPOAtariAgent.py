from agents.PPOAgent import PPOAgent
from algorithms.ReplayBuffer import PPOTrajectoryBuffer
from modules.dop_models.DOPModelAtari import DOPControllerAtari
from modules.PPO_AtariModules import PPOAtariNetworkFM, PPOAtariNetwork, PPOAtariNetworkRND, PPOAtariNetworkQRND, PPOAtariNetworkDOP, PPOAtariMotivationNetwork, PPOAtariNetworkSRRND
from motivation.DOPMotivation import DOPMotivation
from motivation.Encoder import Encoder
from motivation.ForwardModelMotivation import ForwardModelMotivation
from motivation.RNDMotivation import RNDMotivation, QRNDMotivation


class PPOAtariAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, config)
        self.network = PPOAtariNetwork(input_shape, action_dim, config, head=action_type).to(config.device)
        self.algorithm = self.init_algorithm(config, self.memory, action_type)


class PPOAtariRNDAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, config)
        self.network = PPOAtariNetworkRND(input_shape, action_dim, config, head=action_type).to(config.device)
        self.motivation = RNDMotivation(self.network.rnd_model, config.motivation_lr, config.motivation_eta, config.device)
        self.algorithm = self.init_algorithm(config, self.memory, action_type, motivation=True)

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state0.cpu(), value.cpu(), action0.cpu(), probs0.cpu(), state1.cpu(), reward.cpu(), mask.cpu())
        indices = self.memory.indices()
        self.algorithm.train(indices)
        self.motivation.train(self.memory, indices)
        if indices is not None:
            self.memory.clear()


class PPOAtariSRRNDAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, config)
        self.network = PPOAtariNetworkSRRND(input_shape, 512, action_dim, config, head=action_type).to(config.device)
        self.encoder = Encoder(self.network.encoder, 0.0001, config.device)
        self.encoder_memory = PPOTrajectoryBuffer(config.trajectory_size, config.batch_size, config.n_env)
        self.motivation = RNDMotivation(self.network.rnd_model, config.motivation_lr, config.motivation_eta, config.device)
        self.algorithm = self.init_algorithm(config, self.memory, action_type, motivation=True)

    def train(self, state0, features0, value, action0, probs0, state1, features1, reward, mask):
        self.memory.add(features0.cpu(), value.cpu(), action0.cpu(), probs0.cpu(), features1.cpu(), reward.cpu(), mask.cpu())
        self.encoder_memory.add(state0.cpu(), value.cpu(), action0.cpu(), probs0.cpu(), state1.cpu(), reward.cpu(), mask.cpu())
        indices = self.memory.indices()
        memory_indices = self.encoder_memory.indices()
        self.algorithm.train(indices)
        self.motivation.train(self.encoder_memory, memory_indices)
        self.encoder.train(self.encoder_memory, memory_indices)
        if indices is not None:
            self.memory.clear()
        if memory_indices is not None:
            self.encoder_memory.clear()

    def get_features(self, state):
        features = self.network.encoder(state).detach()

        return features



class PPOAtariQRNDAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, config)
        self.network = PPOAtariNetworkQRND(input_shape, action_dim, config, head=action_type).to(config.device)
        self.motivation = QRNDMotivation(self.network.qrnd_model, config.motivation_lr, config.motivation_eta, config.device)
        self.algorithm = self.init_algorithm(config, self.memory, action_type, motivation=True)

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state0.cpu(), value.cpu(), action0.cpu(), probs0.cpu(), state1.cpu(), reward.cpu(), mask.cpu())
        indices = self.memory.indices()
        self.algorithm.train(indices)
        self.motivation.train(self.memory, indices)
        if indices is not None:
            self.memory.clear()


class PPOAtariDOPControllerAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, config)
        self.network = PPOAtariMotivationNetwork(input_shape, action_dim, config, action_type).to(config.device)
        self.algorithm = self.init_algorithm(config, self.memory, action_type, motivation=True)

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state0.cpu(), value.cpu(), action0.cpu(), probs0.cpu(), state1.cpu(), reward.cpu(), mask.cpu())
        indices = self.memory.indices()
        self.algorithm.train(indices)
        if indices is not None:
            self.memory.clear()


class PPOAtariDOPAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type, controller):
        super().__init__(input_shape, action_dim, config)

        self.network = PPOAtariNetworkDOP(input_shape, action_dim, config, action_type, controller.network).to(config.device)
        self.motivation = DOPMotivation(self.network.dop_model, config.motivation_lr, config.lr, config.motivation_eta, config.device)
        self.algorithm = self.init_algorithm(config, self.memory, action_type)
        self.controller = controller

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state0.cpu(), value.cpu(), action0.cpu(), probs0.cpu(), state1.cpu(), reward.cpu(), mask.cpu())
        indices = self.memory.indices()
        self.algorithm.train(indices)
        self.motivation.train(self.memory, indices, self.memory, indices, self.config.batch_size // self.config.dop_heads)
        if indices is not None:
            self.memory.clear()

    def get_action(self, state):
        value, action, probs = self.network(state)

        return value.detach(), action, probs.detach()


class PPOAtariForwardModelAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, config)
        self.network = PPOAtariNetworkFM(input_shape, action_dim, config, head=action_type).to(config.device)
        self.motivation = ForwardModelMotivation(self.network.forward_model, config.forward_model_lr, config.forward_model_eta, config.forward_model_variant, self.memory, config.device)
        self.algorithm = self.init_algorithm(config, self.memory, action_type, motivation=True)

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state0.cpu(), value.cpu(), action0.cpu(), probs0.cpu(), state1.cpu(), reward.cpu(), mask.cpu())
        indices = self.memory.indices()
        self.algorithm.train(indices)
        self.motivation.train(indices)
        if indices is not None:
            self.memory.clear()
