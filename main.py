import gym
import torch

from DDPG import DDPG

if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')

    state0 = torch.tensor([env.reset()], dtype=torch.float32)
    done = False
    total_rewards = 0

    agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0], 1024, 4, 1e-4, 1e-3, 0.99, 1e-3, 1e-2)

    while not done:
        action0 = agent.get_action(state0)
        # env.render()
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action0.detach().numpy()[0])

        state1 = torch.tensor([next_state], dtype=torch.float32)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)

        agent.train(state0, action0, state1, reward, done)

        total_rewards += reward
        state0 = state1

    env.close()
    print(total_rewards)

'''
    

    for i in range(15):
        memory.add(i)
        print(memory.sample(5))
'''
