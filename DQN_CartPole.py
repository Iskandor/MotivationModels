import gym
import torch

from DQN import DQN, DQNCritic
from DiscreteExploration import DiscreteExploration


class Critic(DQNCritic):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__(state_dim, action_dim)
        hidden = 64
        limit = 0.001

        self._hidden0 = torch.nn.Linear(state_dim, hidden)
        self._hidden1 = torch.nn.Linear(hidden, int(hidden / 2))
        self._output = torch.nn.Linear(int(hidden / 2), action_dim)

        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -limit, limit)

    def forward(self, state):
        x = torch.nn.functional.relu(self._hidden0(state))
        x = torch.nn.functional.relu(self._hidden1(x))
        value = self._output(x)
        return value


def test(env, agent):
    state0 = torch.tensor(env.reset(), dtype=torch.float32)
    done = False
    total_rewards = 0

    while not done:
        #env.render()
        action = agent.get_action(state0)
        next_state, reward, done, info = env.step(action)
        total_rewards += reward
        state0 = torch.tensor(next_state, dtype=torch.float32)
    #env.render()
    env.close()
    return total_rewards


def run():
    epochs = 1500

    env = gym.make('CartPole-v0')

    agent = DQN(Critic, env.observation_space.shape[0], env.action_space.n, 10000, 64, 1e-3, 0.99, 1e-2)
    exploration = DiscreteExploration(0.3)

    rewards = torch.zeros(100, dtype=torch.float32)
    reward_index = 0

    for e in range(epochs):
        state0 = torch.tensor(env.reset(), dtype=torch.float32)
        done = False
        rewards[reward_index] = test(env, agent)
        reward_index += 1
        if reward_index == 100:
            reward_index = 0

        while not done:
            # env.render()
            action0 = exploration.explore(agent.get_action(state0), env)
            next_state, reward, done, info = env.step(action0)
            state1 = torch.tensor(next_state, dtype=torch.float32)

            agent.train(state0, action0, state1, reward, done)
            state0 = state1

        exploration.update(e)
        avg_reward = rewards.sum() / 100
        print('Episode ' + str(e) + ' reward ' + str(avg_reward.item()))

    env.close()
