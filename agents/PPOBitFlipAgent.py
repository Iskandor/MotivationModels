from agents.PPOAgent import PPOAgent
from modules.PPO_BitFlipModules import PPOBitFlipNetwork, PPOBitFlipNetworkRND
from motivation.RNDMotivation import RNDMotivation


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
