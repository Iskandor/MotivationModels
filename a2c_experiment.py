import gym
import torch


class ExperimentA2C:
    def __init__(self, env_name, env, config):
        self._env_name = env_name
        self._env = env
        self._config = config
        self._agent = None

    def run_baseline(self, agent, trial):
        config = self._config

        env = gym.make('CartPole-v0')

        rewards = torch.zeros(100, dtype=torch.float32)
        reward_index = 0

        for e in range(config.episodes):
            state0 = torch.tensor(env.reset(), dtype=torch.float32)
            done = False
            total_reward = 0

            while not done:
                # env.render()
                action0, log_prob = agent.get_action(state0)
                next_state, reward, done, info = env.step(action0)
                total_reward += reward
                agent.train(state0, log_prob, reward, done)
                state1 = torch.tensor(next_state, dtype=torch.float32)
                state0 = state1

            rewards[reward_index] = total_reward
            reward_index += 1
            if reward_index == 100:
                reward_index = 0

            avg_reward = rewards.sum() / 100
            print('Episode ' + str(e) + ' reward ' + str(avg_reward.item()))

        env.close()
