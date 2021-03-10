import torch
from torch.distributions import Categorical, Normal

from agents import TYPE
from algorithms.PPO import PPO
from modules.PPO_Modules import PPOSimpleNetwork, PPOAerisNetwork


class PPOAgent:
    def __init__(self, network, state_dim, action_dim, config, action_type, n_env=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network = network

        if action_type == TYPE.discrete:
            self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size, config.beta, config.gamma,
                                 self.log_prob_discrete, self.entropy_discrete, n_env=n_env)
        if action_type == TYPE.continuous:
            self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size, config.beta, config.gamma,
                                 self.log_prob_continuous, self.entropy_continuous, n_env=n_env)
        if action_type == TYPE.multibinary:
            self.algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size, config.beta, config.gamma,
                                 self.log_prob_discrete, self.entropy_discrete, n_env=n_env)

        self.action_type = action_type

    def get_action(self, state):
        value, action, probs = self.network(state)

        return value.detach(), action, probs.detach()

    def convert_action(self, action):
        if self.action_type == TYPE.discrete:
            return action.squeeze(0).item()
        if self.action_type == TYPE.continuous:
            return action.squeeze(0).numpy()
        if self.action_type == TYPE.multibinary:
            return action.squeeze(0).item()

    def train(self, state0, value, action0, probs0, state1, reward, mask):
        self.algorithm.train(state0, value, action0, probs0, state1, reward, mask)

    def train_n_env(self, state0, value, action0, probs0, state1, reward, mask):
        self.algorithm.train_n_env(state0, value, action0, probs0, state1, reward, mask)

    def save(self, path):
        torch.save(self.network.state_dict(), path + '.pth')

    def load(self, path):
        self.network.load_state_dict(torch.load(path + '.pth'))

    @staticmethod
    def log_prob_discrete(probs, actions):
        dist = Categorical(probs)
        log_prob = dist.log_prob(actions.squeeze(1)).unsqueeze(1)

        return log_prob

    @staticmethod
    def entropy_discrete(probs):
        dist = Categorical(probs)
        entropy = -dist.entropy()
        return entropy.mean()

    def log_prob_continuous(self, probs, actions):
        mu, var = probs[:, 0:self.action_dim], probs[:, self.action_dim:self.action_dim*2]
        dist = Normal(mu, var.sqrt())
        log_prob = dist.log_prob(actions)

        return log_prob

    def entropy_continuous(self, probs):
        mu, var = probs[:, 0:self.action_dim], probs[:, self.action_dim:self.action_dim*2]
        dist = Normal(mu, var.sqrt())
        entropy = -dist.entropy()

        return entropy.mean()


class PPOSimpleAgent(PPOAgent):
    def __init__(self, state_dim, action_dim, config, action_type, n_env=1):
        network = PPOSimpleNetwork(state_dim, action_dim, config, head=action_type)
        super().__init__(network, state_dim, action_dim, config, action_type, n_env)


class PPOAerisAgent(PPOAgent):
    def __init__(self, input_shape, action_dim, config, action_type, n_env=1):
        network = PPOAerisNetwork(input_shape, action_dim, config, head=action_type)
        super().__init__(network, input_shape, action_dim, config, action_type, n_env)
