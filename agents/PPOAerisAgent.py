from agents import TYPE
from agents.PPOAgent import PPOAgent
from modules.PPO_AerisModules import PPOAerisNetwork, PPOAerisNetworkRND, PPOAerisNetworkDOP, PPOAerisNetworkDOPRef, PPOAerisGridNetwork
from motivation.DOPMotivation import DOPMotivation
from motivation.RNDMotivation import RNDMotivation


class PPOAerisAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config):
        super().__init__(input_shape, action_dim, config)
        self.network = PPOAerisNetwork(input_shape, action_dim, config).to(config.device)
        self.algorithm = self.init_algorithm(config, self.memory, TYPE.continuous)


class PPOAerisRNDAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config):
        super().__init__(input_shape, action_dim, config)
        self.network = PPOAerisNetworkRND(input_shape, action_dim, config).to(config.device)
        self.motivation = RNDMotivation(self.network.rnd_model, config.forward_model_lr, config.motivation_eta, self.memory, config.device)
        self.algorithm = self.init_algorithm(config, self.memory, TYPE.continuous, self.motivation)

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state0.cpu(), value.cpu(), action0.cpu(), probs0.cpu(), state1.cpu(), reward.cpu(), mask.cpu())
        indices = self.memory.indices()
        self.algorithm.train(indices)
        self.motivation.train(self.memory, indices)
        if indices is not None:
            self.memory.clear()


class PPOAerisDOPAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config):
        super().__init__(input_shape, action_dim, config)
        self.network = PPOAerisNetworkDOP(input_shape, action_dim, config).to(config.device)
        self.motivation = DOPMotivation(self.network.dop_model, config.forward_model_lr, config.forward_model_lr, config.motivation_eta, config.device)
        self.algorithm = self.init_algorithm(config, self.memory, TYPE.continuous)

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state0.cpu(), value.cpu(), action0.cpu(), probs0.cpu(), state1.cpu(), reward.cpu(), mask.cpu())
        indices = self.memory.indices()
        self.algorithm.train(indices)
        self.motivation.train(self.memory, indices, self.memory, indices)
        if indices is not None:
            self.memory.clear()


class PPOAerisDOPRefAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config):
        super().__init__(input_shape, action_dim, config)
        self.network = PPOAerisNetworkDOPRef(input_shape, action_dim, config).to(config.device)
        self.algorithm = self.init_algorithm(config, self.memory, TYPE.continuous)

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state0.cpu(), value.cpu(), action0.cpu(), probs0.cpu(), state1.cpu(), reward.cpu(), mask.cpu())
        indices = self.memory.indices()
        self.algorithm.train(indices)
        if indices is not None:
            self.memory.clear()


class PPOAerisGridAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config):
        super().__init__(input_shape, action_dim, config)
        self.network = PPOAerisGridNetwork(input_shape, action_dim, config).to(config.device)
        self.algorithm = self.init_algorithm(config, self.memory, TYPE.discrete)
