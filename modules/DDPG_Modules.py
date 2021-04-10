import copy

import torch
from torch import nn

from modules.encoders.EncoderAeris import EncoderAeris
from modules.forward_models.ForwardModelAeris import ForwardModelAeris, ForwardModelEncoderAeris
from modules.forward_models.ForwardModelBullet import ForwardModelBullet
from modules.inverse_models.InverseModelAeris import InverseModelAeris
from modules.metacritic_models.MetaCriticModelAeris import MetaCriticModelAeris, MetaCriticRNDModelAeris
from modules.metacritic_models.MetaCriticModelBullet import MetaCriticModelBullet
from modules.rnd_models.RNDModelAeris import RNDModelAeris


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(Critic, self).__init__()

        self._hidden0 = nn.Linear(state_dim, config.critic_h1)
        self._hidden1 = nn.Linear(config.critic_h1 + action_dim, config.critic_h2)
        self._output = nn.Linear(config.critic_h2, 1)

        self.init()
        self.heads = 1

    def forward(self, features, action):
        x = torch.relu(self._hidden0(features))
        x = torch.cat([x, action], 1)
        x = torch.relu(self._hidden1(x))
        value = self._output(x)
        return value

    def init(self):
        nn.init.xavier_uniform_(self._hidden0.weight)
        nn.init.xavier_uniform_(self._hidden1.weight)
        nn.init.uniform_(self._output.weight, -3e-3, 3e-3)


class Critic2Heads(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(Critic2Heads, self).__init__()

        self._hidden0 = nn.Linear(state_dim, config.critic_h1)
        self._hidden1 = nn.Linear(config.critic_h1 + action_dim, config.critic_h2)
        self._output_ve = nn.Linear(config.critic_h2, 1)
        self._output_vi = nn.Linear(config.critic_h2, 1)

        self.init()
        self.heads = 2

    def forward(self, state, action):
        x = state
        x = torch.relu(self._hidden0(x))
        x = torch.cat([x, action], 1)
        x = torch.relu(self._hidden1(x))
        return self._output_ve(x), self._output_vi(x)

    def init(self):
        nn.init.xavier_uniform_(self._hidden0.weight)
        nn.init.xavier_uniform_(self._hidden1.weight)
        nn.init.uniform_(self._output_ve.weight, -3e-3, 3e-3)
        nn.init.uniform_(self._output_vi.weight, -3e-3, 3e-3)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(Actor, self).__init__()

        self._hidden0 = nn.Linear(state_dim, config.actor_h1)
        self._hidden1 = nn.Linear(config.actor_h1, config.actor_h2)
        self._output = nn.Linear(config.actor_h2, action_dim)

        self.init()

    def forward(self, features):
        x = torch.relu(self._hidden0(features))
        x = torch.relu(self._hidden1(x))
        policy = torch.tanh(self._output(x))
        return policy

    def init(self):
        nn.init.xavier_uniform_(self._hidden0.weight)
        nn.init.xavier_uniform_(self._hidden1.weight)
        nn.init.uniform_(self._output.weight, -3e-1, 3e-1)


class DDPGNetwork(nn.Module):
    def __init__(self):
        super(DDPGNetwork, self).__init__()
        self.critic = None
        self.actor = None
        self.critic_target = None
        self.actor_target = None

    def value(self, state, action):
        x = torch.cat([state, action], dim=1)
        value = self.critic(x)
        return value

    def action(self, state):
        policy = self.actor(state)
        return policy

    def value_target(self, state, action):
        x = torch.cat([state, action], dim=1)
        value = self.critic_target(x)
        return value

    def action_target(self, state):
        policy = self.actor_target(state)
        return policy

    def soft_update(self, tau):
        self._soft_update(self.critic_target, self.critic, tau)
        self._soft_update(self.actor_target, self.actor, tau)

    def hard_update(self):
        self._hard_update(self.critic_target, self.critic)
        self._hard_update(self.actor_target, self.actor)

    @staticmethod
    def _soft_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    @staticmethod
    def _hard_update(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


class DDPGSimpleNetwork(DDPGNetwork):
    def __init__(self, input_shape, action_dim, config):
        super(DDPGSimpleNetwork, self).__init__()
        self.critic = Critic(input_shape, action_dim, config)
        self.actor = Actor(input_shape, action_dim, config)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
        self.hard_update()


class DDPGBulletNetwork(DDPGNetwork):
    def __init__(self, state_dim, action_dim, config):
        super(DDPGBulletNetwork, self).__init__()

        critic_h = [int(x) for x in config.critic_h.split(',')]

        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, critic_h[0]),
            nn.ReLU(),
            nn.Linear(critic_h[0], critic_h[1]),
            nn.ReLU(),
            nn.Linear(critic_h[1], 1))

        nn.init.xavier_uniform_(self.critic[0].weight)
        nn.init.xavier_uniform_(self.critic[2].weight)
        nn.init.uniform_(self.critic[4].weight, -0.003, 0.003)

        actor_h = [int(x) for x in config.actor_h.split(',')]

        self.actor = nn.Sequential(
            nn.Linear(state_dim, actor_h[0]),
            nn.ReLU(),
            nn.Linear(actor_h[0], actor_h[1]),
            nn.ReLU(),
            nn.Linear(actor_h[1], action_dim),
            nn.Tanh())

        nn.init.xavier_uniform_(self.actor[0].weight)
        nn.init.xavier_uniform_(self.actor[2].weight)
        nn.init.uniform_(self.actor[4].weight, -0.3, 0.3)

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
        self.hard_update()


class DDPGBulletNetworkFM(DDPGBulletNetwork):
    def __init__(self, state_dim, action_dim, config):
        super(DDPGBulletNetworkFM, self).__init__(state_dim, action_dim, config)
        self.forward_model = ForwardModelBullet(state_dim, action_dim, config)


class DDPGBulletNetworkSU(DDPGBulletNetworkFM):
    def __init__(self, state_dim, action_dim, config):
        super(DDPGBulletNetworkSU, self).__init__(state_dim, action_dim, config)
        self.metacritic_model = MetaCriticModelBullet(self.forward_model, state_dim, action_dim, config)


class DDPGAerisNetwork(DDPGNetwork):
    def __init__(self, input_shape, action_dim, config):
        super(DDPGAerisNetwork, self).__init__()

        self.channels = input_shape[0]
        self.width = input_shape[1]

        fc_count = config.critic_kernels_count * self.width // 4

        self.critic = nn.Sequential(
            nn.Conv1d(self.channels + action_dim, config.critic_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_count, config.critic_h1),
            nn.ReLU(),
            nn.Linear(config.critic_h1, 1))

        nn.init.xavier_uniform_(self.critic[0].weight)
        nn.init.xavier_uniform_(self.critic[3].weight)
        nn.init.uniform_(self.critic[5].weight, -0.003, 0.003)

        fc_count = config.actor_kernels_count * self.width // 4

        self.actor = nn.Sequential(
            nn.Conv1d(self.channels, config.actor_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_count, config.actor_h1),
            nn.ReLU(),
            nn.Linear(config.actor_h1, action_dim),
            nn.Tanh())

        nn.init.xavier_uniform_(self.actor[0].weight)
        nn.init.xavier_uniform_(self.actor[3].weight)
        nn.init.uniform_(self.actor[5].weight, -0.3, 0.3)

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
        self.hard_update()

    def value(self, state, action):
        a = action.unsqueeze(state.ndim - 1).repeat(1, 1, state.shape[2])
        x = torch.cat([state, a], dim=1)
        value = self.critic(x)
        return value

    def value_target(self, state, action):
        a = action.unsqueeze(state.ndim - 1).repeat(1, 1, state.shape[2])
        x = torch.cat([state, a], dim=1)
        value = self.critic_target(x)
        return value


class DDPGAerisNetworkFM(DDPGAerisNetwork):
    def __init__(self, input_shape, action_dim, config):
        super(DDPGAerisNetworkFM, self).__init__(input_shape, action_dim, config)
        self.forward_model = ForwardModelAeris(input_shape, action_dim, config)


class DDPGAerisNetworkFME(DDPGAerisNetwork):
    def __init__(self, input_shape, action_dim, config):
        super(DDPGAerisNetworkFME, self).__init__(input_shape, action_dim, config)
        self.encoder = EncoderAeris(input_shape, action_dim, config)
        self.forward_model = ForwardModelEncoderAeris(self.encoder, action_dim, config, encoder_loss=True)


class DDPGAerisNetworkIM(DDPGAerisNetwork):
    def __init__(self, input_shape, action_dim, config):
        super(DDPGAerisNetworkIM, self).__init__(input_shape, action_dim, config)
        self.encoder = EncoderAeris(input_shape, action_dim, config)
        self.inverse_model = InverseModelAeris(self.encoder, action_dim, config)


class DDPGAerisNetworkFIM(DDPGAerisNetwork):
    def __init__(self, input_shape, action_dim, config):
        super(DDPGAerisNetworkFIM, self).__init__(input_shape, action_dim, config)
        self.encoder = EncoderAeris(input_shape, action_dim, config)
        self.forward_model = ForwardModelEncoderAeris(self.encoder, action_dim, config, encoder_loss=False)
        self.inverse_model = InverseModelAeris(self.encoder, action_dim, config)


class DDPGAerisNetworkSU(DDPGAerisNetwork):
    def __init__(self, input_shape, action_dim, config):
        super(DDPGAerisNetworkSU, self).__init__(input_shape, action_dim, config)
        self.forward_model = ForwardModelAeris(input_shape, action_dim, config)
        self.metacritic_model = MetaCriticModelAeris(self.forward_model, input_shape, action_dim)


class DDPGAerisNetworkM2(DDPGAerisNetwork):
    def __init__(self, input_shape, action_dim, config):
        super(DDPGAerisNetworkM2, self).__init__(input_shape, action_dim, config)

        self.encoder = EncoderAeris(input_shape, action_dim, config)
        self.forward_model = ForwardModelEncoderAeris(self.encoder, action_dim, config, encoder_loss=True)
        self.gate = DDPGSimpleNetwork(2, 2, config)

    def weight(self, gate_state):
        weight = self.gate.action(gate_state)
        weight = nn.functional.softmax(weight, dim=1)

        return weight


class DDPGAerisNetworkRND(DDPGAerisNetwork):
    def __init__(self, input_shape, action_dim, config):
        super(DDPGAerisNetworkRND, self).__init__(input_shape, action_dim, config)
        self.rnd_model = RNDModelAeris(input_shape, action_dim, config)


class DDPGAerisNetworkSURND(DDPGAerisNetwork):
    def __init__(self, input_shape, action_dim, config):
        super(DDPGAerisNetworkSURND, self).__init__(input_shape, action_dim, config)
        self.rnd_model = RNDModelAeris(input_shape, action_dim, config)
        self.metacritic_model = MetaCriticRNDModelAeris(self.rnd_model, input_shape, action_dim, config)
