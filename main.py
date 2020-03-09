import DDPG_MountainCar

if __name__ == '__main__':
    DDPG_MountainCar.run_baseline()
    DDPG_MountainCar.run_forward_model()
    DDPG_MountainCar.run_metalearner_model()
    #DQN_FrozenLake.run()
    #DQN_CartPole.run()
