from gym_bitflip.envs import BitFlipEnv

if __name__ == '__main__':
    env = BitFlipEnv()

    done = False
    state = env.reset()

    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)

        print(reward)
