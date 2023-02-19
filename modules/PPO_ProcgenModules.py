import torch
import torch.nn as nn
import numpy as np

from modules import init_orthogonal
from modules.PPO_Modules import DiscreteHead, Actor, Critic2Heads, ActorNHeads, CriticHead, Critic2NHeads
from modules.encoders.EncoderProcgen import ST_DIMEncoderProcgen
from modules.forward_models.ForwardModelProcgen import ForwardModelProcgen, FWDModelProcgen, ICMModelProcgen
from modules.rnd_models.RNDModelProcgen import VICRegModelProcgen, QRNDModelProcgen, RNDModelProcgen, FEDRefModelProcgen, CNDModelProcgen, BarlowTwinsModelProcgen, CNDVModelProcgen, VINVModelProcgen


class PPOProcgenNetwork(torch.nn.Module):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOProcgenNetwork, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim
        input_channels = self.input_shape[0]
        input_height = self.input_shape[1]
        input_width = self.input_shape[2]
        self.feature_dim = 512

        fc_inputs_count = 64 * (input_width // 8) * (input_height // 8)

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_inputs_count, self.feature_dim)
        )

        init_orthogonal(self.features[0], np.sqrt(2))
        init_orthogonal(self.features[2], np.sqrt(2))
        init_orthogonal(self.features[4], np.sqrt(2))
        init_orthogonal(self.features[6], np.sqrt(2))
        init_orthogonal(self.features[9], np.sqrt(2))

        self.critic = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, 1)
        )

        init_orthogonal(self.critic[1], 0.1)
        init_orthogonal(self.critic[3], 0.01)

        self.actor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            DiscreteHead(self.feature_dim, action_dim)
        )

        init_orthogonal(self.actor[1], 0.01)
        init_orthogonal(self.actor[3], 0.01)

        self.actor = Actor(self.actor, head, self.action_dim)

    def forward(self, state):
        features = self.features(state)
        value = self.critic(features)
        action, probs = self.actor(features)
        action = self.actor.encode_action(action)

        return value, action, probs


class PPOProcgenMotivationNetwork(PPOProcgenNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOProcgenMotivationNetwork, self).__init__(input_shape, action_dim, config, head)

        self.critic = nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.feature_dim),
            torch.nn.ReLU(),
            Critic2Heads(self.feature_dim)
        )

        init_orthogonal(self.critic[0], 0.1)
        init_orthogonal(self.critic[2], 0.01)


class PPOProcgenSRMotivationNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, config, head):
        super(PPOProcgenSRMotivationNetwork, self).__init__()

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


class PPOProcgenNetworkFM(PPOProcgenMotivationNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOProcgenNetworkFM, self).__init__(input_shape, action_dim, config, head)
        self.forward_model = ForwardModelProcgen(self.features, self.feature_dim, self.action_dim)


class PPOProcgenNetworkRND(PPOProcgenMotivationNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOProcgenNetworkRND, self).__init__(input_shape, action_dim, config, head)
        self.rnd_model = RNDModelProcgen(input_shape, self.action_dim, config)


class PPOProcgenNetworkFWD(PPOProcgenMotivationNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOProcgenNetworkFWD, self).__init__(input_shape, action_dim, config, head)
        self.forward_model = FWDModelProcgen(input_shape, 512, self.action_dim, config)


class PPOProcgenNetworkICM(PPOProcgenMotivationNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOProcgenNetworkICM, self).__init__(input_shape, action_dim, config, head)
        self.forward_model = ICMModelProcgen(input_shape, 512, self.action_dim, config)


class PPOProcgenNetworkSRRND(PPOProcgenSRMotivationNetwork):
    def __init__(self, input_shape, feature_dim, action_dim, config, head):
        super(PPOProcgenNetworkSRRND, self).__init__(feature_dim, action_dim, config, head)
        self.encoder = ST_DIMEncoderProcgen(input_shape, feature_dim, config)
        self.rnd_model = RNDModelProcgen(input_shape, action_dim, config)


class PPOProcgenNetworkFEDRef(PPOProcgenMotivationNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOProcgenNetworkFEDRef, self).__init__(input_shape, action_dim, config, head)
        self.fed_ref_model = FEDRefModelProcgen(input_shape, action_dim, self.features, config)


class PPOProcgenNetworkCND(PPOProcgenMotivationNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOProcgenNetworkCND, self).__init__(input_shape, action_dim, config, head)

        if config.type == 'bt':
            self.cnd_model = BarlowTwinsModelProcgen(input_shape, action_dim, config)
        if config.type == 'vicreg':
            self.cnd_model = VICRegModelProcgen(input_shape, action_dim, config)
        if config.type == 'st-dim':
            self.cnd_model = CNDModelProcgen(input_shape, action_dim, config)
        if config.type == 'vanilla':
            self.cnd_model = CNDVModelProcgen(input_shape, action_dim, config)
        if config.type == 'vinv':
            self.cnd_model = VINVModelProcgen(input_shape, action_dim, config)


class PPOProcgenNetworkQRND(PPOProcgenMotivationNetwork):
    def __init__(self, input_shape, action_dim, config, head):
        super(PPOProcgenNetworkQRND, self).__init__(input_shape, action_dim, config, head)
        self.qrnd_model = QRNDModelProcgen(input_shape, action_dim, config)
