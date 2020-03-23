import numpy

import DDPG_MountainCar

import gym  # open ai gym
import pybulletgym

if __name__ == '__main__':
    '''
    env = gym.make('AntPyBulletEnv-v0')
    # env.render() # call this before env.reset, if you want a window showing the environment
    state = env.reset()  # should return a state vector if everything worked
    env.render('human')
    done = False
    steps = 0

    while not done:
        action = (numpy.random.random(env.action_space.shape[0]) - 0.5) * 2
        next_state, reward, done, info = env.step(action)
        steps += 1

    print(steps)
    '''

    #DDPG_MountainCar.run_baseline()
    #DDPG_MountainCar.run_forward_model()
    DDPG_MountainCar.run_metalearner_model()
    #DQN_FrozenLake.run()
    #DQN_CartPole.run()
