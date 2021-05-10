import torch
from torch import nn
from torch.distributions import Categorical, Normal

from agents import TYPE
from algorithms.PPO import PPO
from algorithms.ReplayBuffer import PPOTrajectoryBuffer
from modules.PPO_Modules import PPOSimpleNetwork, PPOAerisNetwork, PPOAtariNetwork, PPOAtariNetworkFM
from motivation.ForwardModelMotivation import ForwardModelMotivation
from utils import one_hot_code


class PPOAgent:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.network = None
        self.memory = PPOTrajectoryBuffer(config.trajectory_size, config.batch_size, config.n_env)
        self.algorithm = None
        self.action_type = None

        # if config.gpus and len(config.gpus) > 1:
        #     config.batch_size *= len(config.gpus)
        #     self.network = nn.DataParallel(self.network, config.gpus)

    def init_algorithm(self, config, memory, action_type, motivation=None):
        self.action_type = action_type
        algorithm = None
        if action_type == TYPE.discrete:
            algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size, memory, config.beta, config.gamma,
                            self.log_prob_discrete, self.entropy_discrete, ppo_epochs=config.ppo_epochs, n_env=config.n_env, device=config.device, motivation=motivation)
        if action_type == TYPE.continuous:
            algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size, memory, config.beta, config.gamma,
                            self.log_prob_continuous, self.entropy_continuous, ppo_epochs=config.ppo_epochs, n_env=config.n_env, device=config.device, motivation=motivation)
        if action_type == TYPE.multibinary:
            algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size, memory, config.beta, config.gamma,
                            self.log_prob_discrete, self.entropy_discrete, ppo_epochs=config.ppo_epochs, n_env=config.n_env, device=config.device, motivation=motivation)

        return algorithm

    def get_action(self, state):
        value, action, probs = self.network(state)

        return value.detach(), self.encode_action(action), probs.detach()

    def convert_action(self, action):
        if self.action_type == TYPE.discrete:
            return torch.argmax(action.squeeze(0)).item()
        if self.action_type == TYPE.continuous:
            return action.squeeze(0).numpy()
        if self.action_type == TYPE.multibinary:
            return action.squeeze(0).item()

    def encode_action(self, action):
        if self.action_type == TYPE.discrete:
            return one_hot_code(action, self.action_dim)
        if self.action_type == TYPE.continuous:
            return action
        if self.action_type == TYPE.multibinary:
            return None  # not implemented

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state0.cpu(), value.cpu(), action0.cpu(), probs0.cpu(), state1.cpu(), reward.cpu(), mask.cpu())
        indices = self.memory.indices()
        self.algorithm.train(indices)
        if indices is not None:
            self.memory.clear()

    def save(self, path):
        torch.save(self.network.state_dict(), path + '.pth')

    def load(self, path):
        self.network.load_state_dict(torch.load(path + '.pth', map_location='cpu'))

    @staticmethod
    def log_prob_discrete(probs, actions):
        actions = torch.argmax(actions, dim=1)
        dist = Categorical(probs)
        log_prob = dist.log_prob(actions).unsqueeze(1)

        return log_prob

    @staticmethod
    def entropy_discrete(probs):
        dist = Categorical(probs)
        entropy = -dist.entropy()
        return entropy.mean()

    def log_prob_continuous(self, probs, actions):
        mu, var = probs[:, 0:self.action_dim], probs[:, self.action_dim:self.action_dim * 2]
        dist = Normal(mu, var.sqrt())
        log_prob = dist.log_prob(actions)

        return log_prob

    def entropy_continuous(self, probs):
        mu, var = probs[:, 0:self.action_dim], probs[:, self.action_dim:self.action_dim * 2]
        dist = Normal(mu, var.sqrt())
        entropy = -dist.entropy()

        return entropy.mean()


class PPOSimpleAgent(PPOAgent):
    def __init__(self, state_dim, action_dim, config, action_type):
        super().__init__(state_dim, action_dim, config)
        self.network = PPOSimpleNetwork(state_dim, action_dim, config, head=action_type).to(config.device)
        self.algorithm = self.init_algorithm(config, self.memory, action_type)


class PPOAerisAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, config)
        self.network = PPOAerisNetwork(input_shape, action_dim, config, head=action_type).to(config.device)
        self.algorithm = self.init_algorithm(config, self.memory, action_type)


class PPOAtariAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, config)
        self.network = PPOAtariNetwork(input_shape, action_dim, config, head=action_type).to(config.device)
        self.algorithm = self.init_algorithm(config, self.memory, action_type)


class PPOAtariForwardModelAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type):
        super().__init__(input_shape, action_dim, config)
        self.network = PPOAtariNetworkFM(input_shape, action_dim, config, head=action_type).to(config.device)
        self.motivation = ForwardModelMotivation(self.network.forward_model, config.forward_model_lr, config.forward_model_eta, config.forward_model_variant, self.memory, config.device)
        self.algorithm = self.init_algorithm(config, self.memory, action_type, self.motivation)

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.memory.add(state0.cpu(), value.cpu(), action0.cpu(), probs0.cpu(), state1.cpu(), reward.cpu(), mask.cpu())
        indices = self.memory.indices()
        self.algorithm.train(indices)
        self.motivation.train(indices)
        if indices is not None:
            self.memory.clear()
