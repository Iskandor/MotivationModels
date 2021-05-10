from enum import Enum

import torch.nn as nn


class ARCH(Enum):
    small_robotic = 0
    robotic = 1
    aeris = 2
    atari = 3


def init(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    layer.bias.data.zero_()
