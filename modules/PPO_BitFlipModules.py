import torch
import torch.nn as nn

from modules.PPO_Modules import Actor, Critic2Heads
from modules.rnd_models.RNDModelBitFlip import RNDModelBitFlip


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
        nn.init.xavier_uniform_(self.critic[0].weight)
        nn.init.xavier_uniform_(self.critic[2].weight)

        self.layers_actor = [
            torch.nn.Linear(state_dim, config.actor_h1),
            nn.ReLU(),
            torch.nn.Linear(config.actor_h1, config.actor_h2),
            nn.ReLU()
        ]
        nn.init.xavier_uniform_(self.layers_actor[0].weight)
        nn.init.xavier_uniform_(self.layers_actor[2].weight)

        self.actor = Actor(action_dim, self.layers_actor, head)

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
        nn.init.xavier_uniform_(self.critic[0].weight)
        nn.init.xavier_uniform_(self.critic[2].weight)

        self.layers_actor = [
            torch.nn.Linear(state_dim, config.actor_h1),
            nn.ReLU(),
            torch.nn.Linear(config.actor_h1, config.actor_h2),
            nn.ReLU()
        ]
        nn.init.xavier_uniform_(self.layers_actor[0].weight)
        nn.init.xavier_uniform_(self.layers_actor[2].weight)

        self.actor = Actor(action_dim, self.layers_actor, head)

    def forward(self, state):
        value = self.critic(state)
        action, probs = self.actor(state)

        return value, action, probs


class PPOBitFlipNetworkRND(PPOBitFlipMotivationNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOBitFlipNetworkRND, self).__init__(input_shape, action_dim, config, head)
        self.rnd_model = RNDModelBitFlip(input_shape, action_dim, config)
