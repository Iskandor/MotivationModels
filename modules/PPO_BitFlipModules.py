import torch
import torch.nn as nn

from modules import init_xavier_uniform
from modules.PPO_Modules import Actor, Critic2Heads, DiscreteHead, ActorNHeads
from modules.rnd_models.RNDModelBitFlip import RNDModelBitFlip, QRNDModelBitFlip, DOPModelBitFlip
from utils import one_hot_code


class PPOBitFlipNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, config, head):
        super(PPOBitFlipNetwork, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(state_dim, config.critic_h1),
            nn.ReLU(),
            nn.Linear(config.critic_h1, config.critic_h2),
            nn.ReLU(),
            torch.nn.Linear(config.critic_h2, 1)
        )
        init_xavier_uniform(self.critic[0])
        init_xavier_uniform(self.critic[2])

        self.layers_actor = nn.Sequential(
            torch.nn.Linear(state_dim, config.actor_h1),
            nn.ReLU(),
            torch.nn.Linear(config.actor_h1, config.actor_h2),
            nn.ReLU(),
            DiscreteHead(config.actor_h2, action_dim)
        )
        init_xavier_uniform(self.layers_actor[0])
        init_xavier_uniform(self.layers_actor[2])

        self.actor = Actor(self.layers_actor, head)

    def forward(self, state):
        value = self.critic(state)
        action, probs = self.actor(state)

        return value, action, probs


class PPOBitFlipMotivationNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, config, head):
        super(PPOBitFlipMotivationNetwork, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(state_dim, config.critic_h1),
            nn.ReLU(),
            nn.Linear(config.critic_h1, config.critic_h2),
            nn.ReLU(),
            Critic2Heads(config.critic_h2)
        )
        init_xavier_uniform(self.critic[0])
        init_xavier_uniform(self.critic[2])

        self.layers_actor = [
            torch.nn.Linear(state_dim, config.actor_h1),
            nn.ReLU(),
            torch.nn.Linear(config.actor_h1, config.actor_h2),
            nn.ReLU(),
            DiscreteHead(config.actor_h2, action_dim)
        ]
        init_xavier_uniform(self.layers_actor[0])
        init_xavier_uniform(self.layers_actor[2])

        self.actor = Actor(self.layers_actor, head)

    def forward(self, state):
        value = self.critic(state)
        action, probs = self.actor(state)

        return value, action, probs


class PPOBitFlipNetworkRND(PPOBitFlipMotivationNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOBitFlipNetworkRND, self).__init__(input_shape, action_dim, config, head)
        self.rnd_model = RNDModelBitFlip(input_shape, action_dim, config)


class PPOBitFlipNetworkQRND(PPOBitFlipMotivationNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOBitFlipNetworkQRND, self).__init__(input_shape, action_dim, config, head)
        self.qrnd_model = QRNDModelBitFlip(input_shape, action_dim, config)


class PPOBitFlipNetworkDOP(PPOBitFlipNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOBitFlipNetworkDOP, self).__init__(input_shape, action_dim, config, head)

        self.state_dim = input_shape
        self.action_dim = action_dim
        self.head_count = config.dop_heads

        self.layers_actor = nn.Sequential(
            torch.nn.Linear(input_shape, config.actor_h1),
            nn.ReLU(),
            ActorNHeads(head, self.head_count, [config.actor_h1, config.actor_h2, action_dim], config)
        )
        init_xavier_uniform(self.layers_actor[0])

        self.actor = Actor(self.layers_actor, head)

        self.motivator = QRNDModelBitFlip(input_shape, action_dim, config)
        self.dop_model = DOPModelBitFlip(self.head_count, input_shape, action_dim, config, None, self.actor, self.motivator)
        self.argmax = None

    def forward(self, state):
        value = self.critic(state)
        action, probs = self.actor(state)
        error = self.dop_model.error(state, action).view(-1, self.head_count).detach()

        action = action.view(-1, self.head_count, 1)
        argmax = error.argmax(dim=1)
        action = action.gather(dim=1, index=argmax.unsqueeze(-1).unsqueeze(-1)).squeeze(1)
        probs = probs.gather(dim=1, index=argmax.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.action_dim)).squeeze(1)

        self.argmax = argmax

        return value, action, probs

    def index(self):
        return self.argmax


