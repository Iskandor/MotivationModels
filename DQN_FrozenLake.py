import gym
import torch

from DQN import DQN, DQNCritic
from DiscreteExploration import DiscreteExploration
from QLearning import QLearning, QCritic


class Critic(QCritic):
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
        value = torch.nn.functional.sigmoid(self._output(x))
        return value


def test_policy(agent):
    env = gym.make('FrozenLake-v0')

    state0 = torch.zeros((env.observation_space.n), dtype=torch.float32)
    policy = '\n'

    for i in range(env.observation_space.n):
        state0.fill_(0)
        state0[i] = 1

        action = agent.get_action(state0)

        if action == 0:
            policy += 'L'
        if action == 1:
            policy += 'D'
        if action == 2:
            policy += 'R'
        if action == 3:
            policy += 'U'

        if (i + 1) % 4 == 0:
            policy += '\n'

    print(policy + '\n')

    for i in range(env.observation_space.n):
        state0.fill_(0)
        state0[i] = 1

        print(agent.activate(state0))


def test(agent):
    env = gym.make('FrozenLake-v0')

    state0 = torch.zeros((env.observation_space.n), dtype=torch.float32)
    state0[env.reset()] = 1
    done = False

    while not done:
        env.render()
        action = agent.get_action(state0)
        next_state, reward, done, info = env.step(action)

        state0.fill_(0)
        state0[next_state] = 1
    env.render()
    env.close()


def run():
    epochs = 5000

    env = gym.make('FrozenLake-v0')

    agent = DQN(Critic, env.observation_space.n, env.action_space.n, 10000, 64, 1e-3, 0.99, 1e-4)
    #agent = QLearning(Critic, env.observation_space.n, env.action_space.n, 1e-3, 0.99)
    exploration = DiscreteExploration(0.9, 0.1, epochs / 2)

    win = 0
    lose = 0

    for e in range(epochs):
        state0 = torch.zeros((env.observation_space.n), dtype=torch.float32)
        state0[env.reset()] = 1
        done = False

        while not done:
            # env.render()
            action = exploration.explore(agent.get_action(state0), env)
            next_state, reward, done, info = env.step(action)

            if done:
                if reward == 1:
                    win += 1
                else:
                    lose += 1

            state1 = torch.zeros((env.observation_space.n), dtype=torch.float32)
            state1[next_state] = 1
            reward = torch.tensor([reward], dtype=torch.float32)
            done = torch.tensor([done], dtype=torch.float32)

            agent.train(state0, action, state1, reward, done)
            state0 = state1

        exploration.update(e)
        print(str(win) + ' / ' + str(lose))

    env.close()

    test(agent)
    test_policy(agent)
