import torch
import torch.nn as nn
import numpy as np

from modules import init_orthogonal
from modules.DOPModelAtari.DOPModelAtari import DOPModelAtari
from modules.PPO_Modules import DiscreteHead, Actor, Critic2Heads, ActorNHeads
from modules.forward_models.ForwardModelAtari import ForwardModelAtari
from modules.rnd_models.RNDModelAtari import QRNDModelAtari, RNDModelAtari


class PPOAtariNetwork(torch.nn.Module):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAtariNetwork, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim
        input_channels = self.input_shape[0]
        input_height = self.input_shape[1]
        input_width = self.input_shape[2]
        self.feature_dim = 512

        fc_inputs_count = 128 * (input_width // 8) * (input_height // 8)

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_inputs_count, self.feature_dim),
            nn.ReLU()
        )

        init_orthogonal(self.features[0], np.sqrt(2))
        init_orthogonal(self.features[2], np.sqrt(2))
        init_orthogonal(self.features[4], np.sqrt(2))
        init_orthogonal(self.features[6], np.sqrt(2))
        init_orthogonal(self.features[9], np.sqrt(2))

        self.critic = nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.feature_dim, 1)
        )

        init_orthogonal(self.critic[0], 0.1)
        init_orthogonal(self.critic[2], 0.01)

        self.actor = nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU(),
            DiscreteHead(self.feature_dim, action_dim)
        )

        init_orthogonal(self.actor[0], 0.01)
        init_orthogonal(self.actor[2], 0.01)

        self.actor = Actor(self.actor, head, self.action_dim)

    def forward(self, state):
        features = self.features(state)
        value = self.critic(features)
        action, probs = self.actor(features)
        action = self.actor.encode_action(action)

        return value, action, probs


class PPOAtariMotivationNetwork(PPOAtariNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAtariMotivationNetwork, self).__init__(input_shape, action_dim, config, head)

        self.critic = nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU(),
            Critic2Heads(self.feature_dim)
        )

        init_orthogonal(self.critic[0], 0.1)
        init_orthogonal(self.critic[2], 0.01)


class PPOAtariNetworkFM(PPOAtariMotivationNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAtariNetworkFM, self).__init__(input_shape, action_dim, config, head)
        self.forward_model = ForwardModelAtari(self.features, self.feature_dim, self.action_dim)


class PPOAtariNetworkRND(PPOAtariMotivationNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAtariNetworkRND, self).__init__(input_shape, action_dim, config, head)
        self.rnd_model = RNDModelAtari(input_shape, self.action_dim, config)


class PPOAtariNetworkQRND(PPOAtariMotivationNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAtariNetworkQRND, self).__init__(input_shape, action_dim, config, head)
        self.qrnd_model = QRNDModelAtari(input_shape, self.action_dim, config)


class PPOAtariNetworkDOP(PPOAtariNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAtariNetworkDOP, self).__init__(input_shape, action_dim, config, head)

        self.head_count = config.dop_heads

        self.actor = nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU(),
            ActorNHeads(head, config.dop_heads, dims=[self.feature_dim, self.feature_dim, action_dim], init='orto')
        )

        init_orthogonal(self.actor[0], 0.01)

        self.actor = Actor(self.actor, head, self.action_dim)

        self.motivator = QRNDModelAtari(input_shape, action_dim, config)
        self.dop_model = DOPModelAtari(config.dop_heads, input_shape, action_dim, config, self.features, self.actor, self.motivator)

    def forward(self, state):
        features = self.features(state)
        value = self.critic(features)
        action, probs = self.actor(features)

        channels = state.shape[1]
        h = state.shape[2]
        w = state.shape[3]

        action = self.actor.encode_action(action.view(-1, 1))
        probs = probs.view(-1, self.action_dim)
        state = state.unsqueeze(1).repeat(1, self.head_count, 1, 1, 1).view(-1, channels, h, w)

        error = self.motivator.error(state, action).view(-1, self.head_count).detach()
        head_index = error.argmax(dim=1)

        action = action.view(-1, self.head_count, self.action_dim)
        probs = probs.view(-1, self.head_count, self.action_dim)

        action = action.gather(dim=1, index=head_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.action_dim)).squeeze(1)
        probs = probs.gather(dim=1, index=head_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.action_dim)).squeeze(1)

        return value, (action, head_index.unsqueeze(-1)), probs