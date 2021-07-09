from enum import Enum

import torch.nn as nn


class ARCH(Enum):
    small_robotic = 0
    robotic = 1
    aeris = 2
    atari = 3


def init_orthogonal(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)


def init_xavier_uniform(layer, gain):
    nn.init.xavier_uniform_(layer.weight, gain)
    nn.init.zeros_(layer.bias)


def init_uniform(layer, gain):
    nn.init.uniform_(layer.weight, -gain, gain)
    nn.init.zeros_(layer.bias)
