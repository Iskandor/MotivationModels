from agents import TYPE
from agents.PPOAgent import PPOAgent
from algorithms.PPO import PPO
from modules.PPO_AerisModules import PPOAerisNetwork, PPOAerisNetworkRND, PPOAerisNetworkDOP, PPOAerisNetworkDOPRef, PPOAerisGridNetwork
from motivation.DOPMotivation import DOPMotivation
from motivation.RNDMotivation import RNDMotivation


class PPOAerisAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config):
        super().__init__(input_shape, action_dim, TYPE.continuous, config)
        self.network = PPOAerisNetwork(input_shape, action_dim, config).to(config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ppo_epochs=config.ppo_epochs, n_env=config.n_env, device=config.device)


class PPOAerisRNDAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config):
        super().__init__(input_shape, action_dim, TYPE.continuous, config)
        self.network = PPOAerisNetworkRND(input_shape, action_dim, config).to(config.device)
        self.motivation = RNDMotivation(self.network.rnd_model, config.forward_model_lr, config.motivation_eta, config.device)
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


class PPOAerisDOPAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config):
        super().__init__(input_shape, action_dim, TYPE.continuous, config)
        self.network = PPOAerisNetworkDOP(input_shape, action_dim, config).to(config.device)
        self.motivation = DOPMotivation(self.network.dop_model, config.forward_model_lr, config.forward_model_lr, config.motivation_eta, config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ppo_epochs=config.ppo_epochs, n_env=config.n_env, device=config.device)

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state=state0.cpu(), value=value.cpu(), action=action0.cpu(), prob=probs0.cpu(), reward=reward.cpu(), mask=mask.cpu())
        indices = self.memory.indices()
        self.algorithm.train(self.memory, indices)
        self.motivation.train(self.memory, indices, self.memory, indices)
        if indices is not None:
            self.memory.clear()


class PPOAerisDOPRefAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config):
        super().__init__(input_shape, action_dim, TYPE.continuous, config)
        self.network = PPOAerisNetworkDOPRef(input_shape, action_dim, config).to(config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ppo_epochs=config.ppo_epochs, n_env=config.n_env, device=config.device)


class PPOAerisGridAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config):
        super().__init__(input_shape, action_dim, TYPE.discrete, config)
        self.network = PPOAerisGridNetwork(input_shape, action_dim, config).to(config.device)
        self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size,
                             config.beta, config.gamma, ppo_epochs=config.ppo_epochs, n_env=config.n_env, device=config.device)
