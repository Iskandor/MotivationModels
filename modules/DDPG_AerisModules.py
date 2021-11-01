import copy

import torch
import torch.nn as nn
from torch.distributions import Categorical

from modules import init_xavier_uniform, init_uniform
from modules.DDPG_Modules import DDPGNetwork, DDPGSimpleNetwork, ActorNHeads
from modules.encoders.EncoderAeris import EncoderAeris
from modules.forward_models.ForwardModelAeris import ForwardModelAeris, ForwardModelEncoderAeris
from modules.inverse_models.InverseModelAeris import InverseModelAeris
from modules.metacritic_models.MetaCriticModelAeris import MetaCriticModelAeris, MetaCriticRNDModelAeris
from modules.rnd_models.RNDModelAeris import RNDModelAeris, QRNDModelAeris, DOPModelAeris, DOPV2ModelAeris, DOPV2QModelAeris, DOPV3ModelAeris, VanillaQRNDModelAeris, VanillaDOPModelAeris, \
    QRNDModelAerisFC


class Critic2Heads(nn.Module):
    def __init__(self, input_shape, action_dim, config):
        super(Critic2Heads, self).__init__()

        self.channels = input_shape[0]
        self.width = input_shape[1]

        fc_count = config.critic_kernels_count * self.width // 4

        self.features = nn.Sequential(
            nn.Conv1d(self.channels + action_dim, config.critic_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Flatten())

        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fc_count, config.critic_h1),
                    nn.ReLU(),
                    nn.Linear(config.critic_h1, 1)),
                nn.Sequential(
                    nn.Linear(fc_count, config.critic_h1),
                    nn.ReLU(),
                    nn.Linear(config.critic_h1, 1)),
            ]
        )

    def forward(self, state, action):
        a = action.unsqueeze(state.ndim - 1).repeat(1, 1, state.shape[2])
        x = torch.cat([state, a], dim=1)
        x = self.features(x)

        val_ext = self.heads[0](x)
        val_int = self.heads[1](x)

        return val_ext, val_int


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

        self._init_critic(self.critic)

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

    @staticmethod
    def _init_critic(critic):
        init_xavier_uniform(critic[0])
        init_xavier_uniform(critic[3])
        init_uniform(critic[5], 0.003)

    def value(self, state, action):
        a = action.unsqueeze(state.ndim - 1).repeat(1, 1, state.shape[2])
        x = torch.cat([state, a], dim=1)
        value = self.critic(x)
        return value

    def value_target(self, state, action):
        with torch.no_grad():
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


class DDPGAerisNetworkQRND(DDPGAerisNetwork):
    def __init__(self, input_shape, action_dim, config):
        super(DDPGAerisNetworkQRND, self).__init__(input_shape, action_dim, config)
        self.qrnd_model = QRNDModelAeris(input_shape, action_dim, config)


class DDPGAerisNetworkVanillaDOP(DDPGAerisNetwork):
    def __init__(self, input_shape, action_dim, config):
        super(DDPGAerisNetworkVanillaDOP, self).__init__(input_shape, action_dim, config)

        self.action_dim = action_dim
        self.channels = input_shape[0]
        self.width = input_shape[1]
        self.head_count = config.dop_heads

        fc_count = config.critic_kernels_count * self.width // 4

        self.actor = nn.Sequential(
            nn.Conv1d(self.channels, config.actor_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_count, fc_count),
            nn.ReLU(),
            ActorNHeads(self.head_count, fc_count, action_dim, config, init='xavier')
        )

        init_xavier_uniform(self.actor[0])
        init_xavier_uniform(self.actor[3])

        # self.motivator = VanillaQRNDModelAeris(input_shape, action_dim, config, init='orto')
        self.motivator = QRNDModelAerisFC(input_shape, action_dim, config, init='orto')
        self.dop_model = VanillaDOPModelAeris(self.head_count, input_shape, action_dim, config, None, self.actor, self.motivator)

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
        self.hard_update()

    def select_action(self, state, action):
        state = state.unsqueeze(1).repeat(1, self.head_count, 1, 1).view(-1, self.channels, self.width)
        action = action.view(-1, self.action_dim)

        error = self.motivator.error(state, action).view(-1, self.head_count).detach()
        action = action.view(-1, self.head_count, self.action_dim)
        argmax = error.argmax(dim=1)
        action = action.gather(dim=1, index=argmax.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.action_dim)).squeeze(1)
        action = torch.cat([action, argmax.unsqueeze(-1)], dim=1)

        return action

    def all_actions(self, state):
        actions = self.actor(state).view(-1, self.action_dim)

        return actions

    def action(self, state):
        action = self.actor(state)
        action = self.select_action(state, action)

        return action

    def action_target(self, state):
        with torch.no_grad():
            action = self.actor_target(state)

            action = self.select_action(state, action)

        return action

    def value(self, state, action):
        action = action[:, :-1]
        a = action.unsqueeze(state.ndim - 1).repeat(1, 1, state.shape[2])
        x = torch.cat([state, a], dim=1)
        value = self.critic(x)
        return value

    def value_target(self, state, action):
        action = action[:, :-1]
        a = action.unsqueeze(state.ndim - 1).repeat(1, 1, state.shape[2])
        x = torch.cat([state, a], dim=1)
        value = self.critic_target(x)
        return value


class DDPGAerisNetworkDOP(DDPGAerisNetwork):
    def __init__(self, input_shape, action_dim, config):
        super(DDPGAerisNetworkDOP, self).__init__(input_shape, action_dim, config)

        self.action_dim = action_dim
        self.channels = input_shape[0]
        self.width = input_shape[1]
        self.head_count = config.dop_heads

        fc_count = config.critic_kernels_count * self.width // 4

        self.critic = nn.ModuleList([copy.deepcopy(self.critic) for _ in range(self.head_count)])

        for c in self.critic:
            DDPGAerisNetworkDOP._init_critic(c)

        self.actor = nn.Sequential(
            nn.Conv1d(self.channels, config.actor_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            ActorNHeads(self.head_count, fc_count, action_dim, config, init='xavier')
        )

        init_xavier_uniform(self.actor[0])

        self.motivator = VanillaQRNDModelAeris(input_shape, action_dim, config)
        self.dop_model = DOPModelAeris(self.head_count, input_shape, action_dim, config, None, self.actor, self.motivator)

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
        self.hard_update()

    def select_action(self, state, action):
        state = state.unsqueeze(1).repeat(1, self.head_count, 1, 1).view(-1, self.channels, self.width)
        action = action.view(-1, self.action_dim)

        error = self.motivator.error(state, action).view(-1, self.head_count).detach()
        action = action.view(-1, self.head_count, self.action_dim)
        argmax = error.argmax(dim=1)
        action = action.gather(dim=1, index=argmax.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.action_dim)).squeeze(1)
        action = torch.cat([action, argmax.unsqueeze(-1)], dim=1)

        return action

    def action(self, state):
        action = self.actor(state)
        action = self.select_action(state, action)

        return action

    def all_actions(self, state):
        actions = self.actor(state).view(-1, self.action_dim)

        return actions

    def action_target(self, state):
        with torch.no_grad():
            action = self.actor_target(state)
        action = self.select_action(state, action)

        return action

    def _value(self, critic, state, action, grad=True):
        index = action[:, -1].type(torch.int32)
        action = action[:, :-1]
        a = action.unsqueeze(state.ndim - 1).repeat(1, 1, state.shape[2])
        x = torch.cat([state, a], dim=1)
        value = []

        x_head = []
        for i in range(self.head_count):
            indices = (index == i).nonzero(as_tuple=False)
            if indices.shape[0] > 0:
                x_head.append(x[indices].squeeze(1))
            else:
                x_head.append([])

        if grad:
            for i in range(self.head_count):
                if len(x_head[i]) > 0:
                    value.append(critic[i](x_head[i]))
        else:
            with torch.no_grad():
                for i in range(self.head_count):
                    if len(x_head[i]) > 0:
                        value.append(critic[i](x_head[i]))

        return torch.cat(value, dim=0)

    def value(self, state, action):
        return self._value(self.critic, state, action)

    def value_target(self, state, action):
        return self._value(self.critic_target, state, action, grad=False)


class DDPGAerisNetworkDOPV2(DDPGAerisNetwork):
    def __init__(self, input_shape, action_dim, config):
        super(DDPGAerisNetworkDOPV2, self).__init__(input_shape, action_dim, config)

        self.action_dim = action_dim
        self.channels = input_shape[0]
        self.width = input_shape[1]
        self.head_count = config.dop_heads
        self.argmax = None
        self.arbiter_accuracy = 0

        fc_count = config.critic_kernels_count * self.width // 4

        self.actor = nn.Sequential(
            nn.Conv1d(self.channels, config.actor_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            ActorNHeads(self.head_count, fc_count, action_dim, config)
        )

        init_xavier_uniform(self.actor[0])

        self.arbiter = nn.Sequential(
            nn.Conv1d(self.channels, config.forward_model_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(config.forward_model_kernels_count, config.forward_model_kernels_count * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(config.forward_model_kernels_count * 2, config.forward_model_kernels_count * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_count, fc_count),
            nn.ReLU(),
            nn.Linear(fc_count, self.head_count)
        )

        init_xavier_uniform(self.arbiter[0])
        init_xavier_uniform(self.arbiter[2])
        init_xavier_uniform(self.arbiter[4])
        init_xavier_uniform(self.arbiter[7])
        init_xavier_uniform(self.arbiter[9])

        self.motivator = QRNDModelAeris(input_shape, action_dim, config)
        self.dop_model = DOPV2ModelAeris(self.head_count, input_shape, action_dim, config, None, self.actor, self.motivator, self.arbiter)

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
        self.hard_update()

    def action(self, state):
        actions = self.actor(state)
        action, pred_argmax = self._select_action(state, actions)
        self.argmax = pred_argmax

        if state.shape[0] == 1:
            state = state.unsqueeze(1).repeat(1, self.head_count, 1, 1).view(-1, self.channels, self.width)
            actions = actions.view(-1, self.action_dim)
            error = self.motivator.error(state, actions).view(-1, self.head_count).detach()
            target_argmax = error.argmax(dim=1)

            self.arbiter_accuracy = 0
            if pred_argmax == target_argmax:
                self.arbiter_accuracy = 1

        return action

    def index(self):
        return self.argmax

    def accuracy(self):
        return self.arbiter_accuracy

    def action_target(self, state):
        with torch.no_grad():
            action = self.actor_target(state)
            action, _ = self._select_action(state, action)

        return action

    def _select_action(self, state, action):
        probs = torch.softmax(self.arbiter(state), dim=1)
        dist = Categorical(probs)
        index = dist.sample()
        return action.gather(dim=1, index=index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.action_dim)).squeeze(1), index


class DDPGAerisNetworkDOPV3(DDPGAerisNetwork):
    def __init__(self, input_shape, action_dim, config):
        super(DDPGAerisNetworkDOPV3, self).__init__(input_shape, action_dim, config)

        self.critic_heads = 2
        self.action_dim = action_dim
        self.channels = input_shape[0]
        self.width = input_shape[1]
        self.head_count = config.dop_heads
        self.argmax = None

        fc_count = config.critic_kernels_count * self.width // 4

        self.actor = nn.Sequential(
            nn.Conv1d(self.channels, config.actor_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            ActorNHeads(self.head_count, fc_count, action_dim, config)
        )

        init_xavier_uniform(self.actor[0])

        self.critic = Critic2Heads(input_shape, action_dim, config)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.hard_update()

        self.motivator = QRNDModelAeris(input_shape, action_dim, config)
        self.dop_model = DOPV3ModelAeris(self.head_count, input_shape, action_dim, config, None, self.actor, self.critic, self.motivator)

    def action(self, state):
        actions = self.actor(state)
        action, pred_argmax = self._select_action(state, actions)
        self.argmax = pred_argmax

        return action

    def index(self):
        return self.argmax

    def value(self, state, action):
        value = self.critic(state, action)
        return value

    def action_target(self, state):
        with torch.no_grad():
            action = self.actor_target(state)
            action, _ = self._select_action(state, action)

        return action

    def value_target(self, state, action):
        with torch.no_grad():
            value = self.critic_target(state, action)
        return value

    def _select_action(self, state, action):
        state = state.unsqueeze(1).repeat(1, self.head_count, 1, 1).view(-1, self.channels, self.width)
        action = action.view(-1, self.action_dim)

        _, value_int = self.critic(state, action)
        value_int = value_int.view(-1, self.head_count).detach()
        action = action.view(-1, self.head_count, self.action_dim)

        probs = torch.softmax(value_int, dim=1)
        dist = Categorical(probs)
        argmax = dist.sample()
        # argmax = value_int.argmax(dim=1)
        action = action.gather(dim=1, index=argmax.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.action_dim)).squeeze(1)

        return action, argmax


class DDPGAerisNetworkDOPV2Q(DDPGAerisNetwork):
    def __init__(self, input_shape, action_dim, config):
        super(DDPGAerisNetworkDOPV2Q, self).__init__(input_shape, action_dim, config)

        self.action_dim = action_dim
        self.channels = input_shape[0]
        self.width = input_shape[1]
        self.head_count = config.dop_heads
        self.argmax = None

        fc_count = config.critic_kernels_count * self.width // 4

        self.layers_actor = [
            nn.Conv1d(self.channels, config.actor_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_count, config.actor_h1),
            nn.ReLU()
        ]

        init_xavier_uniform(self.layers_actor[0])
        init_xavier_uniform(self.layers_actor[3])

        self.arbiter = nn.Sequential(
            nn.Conv1d(self.channels, config.actor_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_count, fc_count),
            nn.ReLU(),
            nn.Linear(fc_count, fc_count),
            nn.ReLU(),
            nn.Linear(fc_count, self.head_count)
        )

        init_xavier_uniform(self.arbiter[0])
        init_xavier_uniform(self.arbiter[3])
        init_xavier_uniform(self.arbiter[5])
        init_xavier_uniform(self.arbiter[7])

        self.actor = ActorNHeads(self.head_count, action_dim, self.layers_actor, config)
        self.motivator = QRNDModelAeris(input_shape, action_dim, config)
        self.dop_model = DOPV2QModelAeris(input_shape, action_dim, config, None, self.actor, self.motivator, self.arbiter)

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
        self.hard_update()

    def action(self, state):
        x = state
        action = self.actor(x)
        argmax = self.arbiter(x).argmax(dim=1)
        action = action.gather(dim=1, index=argmax.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.action_dim))

        self.argmax = argmax

        return action.squeeze(1)

    def index(self):
        return self.argmax

    def action_target(self, state):
        with torch.no_grad():
            action = self.actor_target(state)
            argmax = self.arbiter(state).argmax(dim=1)
            action = action.gather(dim=1, index=argmax.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.action_dim))

        return action.squeeze(1)


class DDPGAerisNetworkDOPRef(DDPGAerisNetwork):
    def __init__(self, input_shape, action_dim, config):
        super(DDPGAerisNetworkDOPRef, self).__init__(input_shape, action_dim, config)

        self.action_dim = action_dim
        self.channels = input_shape[0]
        self.width = input_shape[1]
        self.head_count = config.dop_heads
        self.device = config.device

        fc_count = config.critic_kernels_count * self.width // 4

        self.actor = nn.Sequential(
            nn.Conv1d(self.channels, config.actor_kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(fc_count, fc_count),
            nn.ReLU(),
            ActorNHeads(self.head_count, fc_count, action_dim, config, init='xavier')
        )

        init_xavier_uniform(self.actor[0])
        init_xavier_uniform(self.actor[3])

        self.argmax = None

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
        self.hard_update()

    def action(self, state):
        x = state
        action = self.actor(x)

        argmax = torch.randint(0, self.head_count, (state.shape[0],), device=self.device)
        action = action.gather(dim=1, index=argmax.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.action_dim))
        self.argmax = argmax

        return action.squeeze(1)

    def index(self):
        return self.argmax

    def action_target(self, state):
        with torch.no_grad():
            action = self.actor_target(state)
            argmax = torch.randint(0, self.head_count, (state.shape[0],), device=self.device)
            action = action.gather(dim=1, index=argmax.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.action_dim))

        return action.squeeze(1)


class DDPGAerisNetworkSURND(DDPGAerisNetwork):
    def __init__(self, input_shape, action_dim, config):
        super(DDPGAerisNetworkSURND, self).__init__(input_shape, action_dim, config)
        self.rnd_model = RNDModelAeris(input_shape, action_dim, config)
        self.metacritic_model = MetaCriticRNDModelAeris(self.rnd_model, input_shape, action_dim, config)
