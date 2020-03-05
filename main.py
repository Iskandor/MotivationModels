import torch

import test

import DDPG_MountainCar
import DQN_CartPole
import DQN_FrozenLake
from ReplayBuffer import ReplayBuffer

if __name__ == '__main__':
    DDPG_MountainCar.run()
    #DQN_FrozenLake.run()
    #DQN_CartPole.run()
