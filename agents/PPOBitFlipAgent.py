from agents.PPOAgent import PPOAgent
from algorithms.ReplayBuffer import MDPTrajectoryBuffer
from modules.PPO_BitFlipModules import PPOBitFlipNetwork, PPOBitFlipNetworkRND, PPOBitFlipNetworkQRND, PPOBitFlipNetworkDOP
from motivation.RNDMotivation import RNDMotivation, QRNDMotivation, DOPMotivation


class PPOBitFlipAgent(PPOAgent):
    def __init__(self, state_dim, action_dim, config, action_type):
        super().__init__(state_dim, action_dim, config)
        self.network = PPOBitFlipNetwork(state_dim, action_dim, config, head=action_type).to(config.device)
        self.algorithm = self.init_algorithm(config, self.memory, action_type)


class PPOBitFlipRNDAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, config)
        self.network = PPOBitFlipNetworkRND(input_shape, action_dim, config, head=action_type).to(config.device)
        self.motivation = RNDMotivation(self.network.rnd_model, config.motivation_lr, config.motivation_eta, config.device)
        self.algorithm = self.init_algorithm(config, self.memory, action_type, self.motivation)

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state0.cpu(), value.cpu(), action0.cpu(), probs0.cpu(), state1.cpu(), reward.cpu(), mask.cpu())
        indices = self.memory.indices()
        self.algorithm.train(indices)
        self.motivation.train(self.memory, indices)
        if indices is not None:
            self.memory.clear()


class PPOBitFlipQRNDAgent(PPOBitFlipRNDAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, config, action_type)
        self.network = PPOBitFlipNetworkQRND(input_shape, action_dim, config, head=action_type).to(config.device)
        self.motivation = QRNDMotivation(self.network.qrnd_model, config.motivation_lr, config.motivation_eta, config.device)
        self.algorithm = self.init_algorithm(config, self.memory, action_type, self.motivation)


class PPOBitFlipDOPAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, config)
        self.network = PPOBitFlipNetworkDOP(input_shape, action_dim, config, head=action_type).to(config.device)
        self.motivation = DOPMotivation(self.network.dop_model, config.motivation_lr, config.motivation_eta, config.device)
        self.algorithm = self.init_algorithm(config, self.memory, action_type, motivation=None)

        self.motivation_memory = MDPTrajectoryBuffer(self.config.forward_model_batch_size, self.config.forward_model_batch_size, config.n_env)

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state0.cpu(), value.cpu(), action0.cpu(), probs0.cpu(), state1.cpu(), reward.cpu(), mask.cpu())
        # self.motivation_memory.add(state0.cpu(), action0.cpu(), state1.cpu(), reward.cpu(), mask.cpu())
        indices = self.memory.indices()
        self.algorithm.train(indices)

        motivator_indices = self.memory.indices()
        generator_indices = self.memory.indices()
        self.motivation.train(self.memory, motivator_indices, self.memory, generator_indices)

        if indices is not None:
            self.memory.clear()

        if generator_indices is not None:
            self.motivation_memory.clear()

    def get_head_index(self):
        return self.network.index()
