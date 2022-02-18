from enum import Enum

import torch
import torch.nn as nn


class ARCH(Enum):
    small_robotic = 0
    robotic = 1
    aeris = 2
    atari = 3


def init_custom(layer, weight_tensor):
    layer.weight = torch.nn.Parameter(torch.clone(weight_tensor))
    nn.init.zeros_(layer.bias)


def init_coupled_orthogonal(layers, gain=1.0):
    weight = torch.zeros(len(layers) * layers[0].weight.shape[0], *layers[0].weight.shape[1:])
    nn.init.orthogonal_(weight, gain)
    weight = weight.reshape(len(layers), *layers[0].weight.shape)

    for i, l in enumerate(layers):
        init_custom(l, weight[i])


def init_orthogonal(layer, gain=1.0):
    __init_general(nn.init.orthogonal_, layer, gain)


def init_xavier_uniform(layer, gain=1.0):
    __init_general(nn.init.xavier_uniform_, layer, gain)


def init_uniform(layer, gain=1.0):
    __init_general(nn.init.uniform_, layer, gain)


def __init_general(function, layer, gain):
    if type(layer.weight) is tuple:
        for w in layer.weight:
            function(w, gain)
    else:
        function(layer.weight, gain)

    if type(layer.bias) is tuple:
        for b in layer.bias:
            nn.init.zeros_(b)
    else:
        nn.init.zeros_(layer.bias)


def init_general_wb(function, weight, bias, gain):
    function(weight, gain)
    nn.init.zeros_(bias)
