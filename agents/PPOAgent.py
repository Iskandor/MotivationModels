import torch

from agents import TYPE
from algorithms.PPO import PPO
from algorithms.ReplayBuffer import PPOTrajectoryBuffer2
from modules.PPO_Modules import PPOSimpleNetwork
from motivation.ForwardModelMotivation import ForwardModelMotivation
from motivation.RNDMotivation import RNDMotivation
from utils import one_hot_code


class PPOAgent:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.network = None
        self.memory = PPOTrajectoryBuffer2(config.trajectory_size, config.batch_size, config.n_env)
        self.algorithm = None
        self.action_type = None

        # if config.gpus and len(config.gpus) > 1:
        #     config.batch_size *= len(config.gpus)
        #     self.network = nn.DataParallel(self.network, config.gpus)

    def init_algorithm(self, config, memory, action_type, motivation=None):
        self.action_type = action_type
        algorithm = PPO(self.network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size, memory, config.beta, config.gamma,
                        ppo_epochs=config.ppo_epochs, n_env=config.n_env, device=config.device, motivation=motivation)

        return algorithm

    def get_action(self, state):
        value, action, probs = self.network(state)

        return value.detach(), self.encode_action(action), probs.detach()

    def convert_action(self, action):
        if self.action_type == TYPE.discrete:
            a = torch.argmax(action, dim=1).numpy()
            return a
        if self.action_type == TYPE.continuous:
            return action.squeeze(0).numpy()
        if self.action_type == TYPE.multibinary:
            return torch.argmax(action, dim=1).numpy()

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


class PPOSimpleAgent(PPOAgent):
    def __init__(self, state_dim, action_dim, config, action_type):
        super().__init__(state_dim, action_dim, config)
        self.network = PPOSimpleNetwork(state_dim, action_dim, config, head=action_type).to(config.device)
        self.algorithm = self.init_algorithm(config, self.memory, action_type)