import torch
import torch.nn as nn
import numpy as np

from modules import init_orthogonal
from modules.dop_models.DOPModelAtari import DOPModelAtari, DOPControllerAtari, DOPActorAtari, DOPGeneratorAtari
from modules.PPO_Modules import DiscreteHead, Actor, Critic2Heads, ActorNHeads, CriticNHeads, Critic2NHeads
from modules.encoders.EncoderAtari import EncoderAtari, AutoEncoderAtari, VAEAtari, DDMEncoderAtari, ST_DIMEncoderAtari
from modules.forward_models.ForwardModelAtari import ForwardModelAtari
from modules.rnd_models.RNDModelAtari import QRNDModelAtari, RNDModelAtari, CNDModelAtari


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


class PPOAtariSRMotivationNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, config, head):
        super(PPOAtariSRMotivationNetwork, self).__init__()

        self.critic = nn.Sequential(
            torch.nn.Linear(state_dim, state_dim),
            torch.nn.ReLU(),
            Critic2Heads(state_dim)
        )

        init_orthogonal(self.critic[0], 0.1)
        init_orthogonal(self.critic[2], 0.01)

        self.actor = nn.Sequential(
            torch.nn.Linear(state_dim, state_dim),
            torch.nn.ReLU(),
            DiscreteHead(state_dim, action_dim)
        )

        init_orthogonal(self.actor[0], 0.01)
        init_orthogonal(self.actor[2], 0.01)

        self.actor = Actor(self.actor, head, action_dim)

    def forward(self, features):
        value = self.critic(features)
        action, probs = self.actor(features)
        action = self.actor.encode_action(action)

        return value, action, probs


class PPOAtariNetworkFM(PPOAtariMotivationNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAtariNetworkFM, self).__init__(input_shape, action_dim, config, head)
        self.forward_model = ForwardModelAtari(self.features, self.feature_dim, self.action_dim)


class PPOAtariNetworkRND(PPOAtariMotivationNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAtariNetworkRND, self).__init__(input_shape, action_dim, config, head)
        self.rnd_model = RNDModelAtari(input_shape, self.action_dim, config)


class PPOAtariNetworkSRRND(PPOAtariSRMotivationNetwork):
    def __init__(self, input_shape, feature_dim, action_dim, config, head):
        super(PPOAtariNetworkSRRND, self).__init__(feature_dim, action_dim, config, head)
        self.encoder = ST_DIMEncoderAtari(input_shape, feature_dim, config)
        self.rnd_model = RNDModelAtari(input_shape, action_dim, config)


class PPOAtariNetworkCND(PPOAtariSRMotivationNetwork):
    def __init__(self, input_shape, feature_dim, action_dim, config, head):
        super(PPOAtariNetworkCND, self).__init__(feature_dim, action_dim, config, head)
        self.encoder = ST_DIMEncoderAtari(input_shape, feature_dim, config)
        self.cnd_model = CNDModelAtari(input_shape, action_dim, config, self.encoder)


class PPOAtariNetworkQRND(PPOAtariSRMotivationNetwork):
    def __init__(self, input_shape, feature_dim, action_dim, config, head):
        super(PPOAtariNetworkQRND, self).__init__(feature_dim, action_dim, config, head)
        self.encoder = ST_DIMEncoderAtari(input_shape, feature_dim, config)
        self.qrnd_model = QRNDModelAtari(input_shape, action_dim, config)


class PPOAtariNetworkDOP(PPOAtariNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAtariNetworkDOP, self).__init__(input_shape, action_dim, config, head)

        self.n_env = config.n_env
        self.head_count = config.dop_heads

        self.encoder = ST_DIMEncoderAtari(input_shape, self.feature_dim, config)

        self.critic = nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU(),
            Critic2NHeads(self.feature_dim, config.dop_heads)
        )

        init_orthogonal(self.critic[0], 0.1)
        init_orthogonal(self.critic[2], 0.01)

        actor = nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU(),
            ActorNHeads(head, config.dop_heads, dims=[self.feature_dim, self.feature_dim, action_dim], init='orto')
        )

        init_orthogonal(actor[0], 0.01)

        self.actor = Actor(actor, head, self.action_dim)

        self.qrnd_model = QRNDModelAtari(input_shape, action_dim, config)
        self.dop_actor = DOPActorAtari(config.dop_heads, input_shape, action_dim, self.actor, self.critic)

        self.dop_controller = DOPControllerAtari(self.feature_dim, 256, config.dop_heads, config)

    def forward(self, features0_0, features0_1):
        value, action, probs = self.dop_actor(features0_0)
        head_value, head_action, head_probs = self.dop_controller(features0_1)

        return features0_0, value, action, probs, features0_1, head_value, head_action, head_probs

class PPOAtariNetworkDOPA(PPOAtariNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOAtariNetworkDOPA, self).__init__(input_shape, action_dim, config, head)

        self.n_env = config.n_env
        self.head_count = config.dop_heads

        self.encoder = ST_DIMEncoderAtari(input_shape, self.feature_dim, config)

        self.critic = nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU(),
            CriticNHeads(self.feature_dim, config.dop_heads)
        )

        init_orthogonal(self.critic[0], 0.1)
        init_orthogonal(self.critic[2], 0.01)

        actor = nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU(),
            ActorNHeads(head, config.dop_heads, dims=[self.feature_dim, self.feature_dim, action_dim], init='orto')
        )

        init_orthogonal(actor[0], 0.01)

        self.actor = Actor(actor, head, self.action_dim)

        self.dop_actor = DOPActorAtari(config.dop_heads, input_shape, action_dim, self.actor, self.critic)

        self.dop_controller = DOPControllerAtari(self.feature_dim, 256, config.dop_heads, config)

    def forward(self, features0_0, features0_1):
        value, action, probs = self.dop_actor(features0_0)
        head_value, head_action, head_probs = self.dop_controller(features0_1)
        return value, action, probs, head_value, head_action, head_probs
