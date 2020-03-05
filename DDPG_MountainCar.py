import gym
import torch

from ContinuousExploration import GaussianExploration, OUExploration
from DDPG import DDPG, DDPGCritic, DDPGActor


class Critic(DDPGCritic):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__(state_dim, action_dim)

        self._hidden0 = torch.nn.Linear(state_dim, 400)
        self._hidden1 = torch.nn.Linear(400 + action_dim, 300)
        self._output = torch.nn.Linear(300, 1)

        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.nn.functional.relu(self._hidden0(state))
        x = torch.cat([x, action], 1)
        x = torch.nn.functional.relu(self._hidden1(x))
        value = self._output(x)
        return value


class Actor(DDPGActor):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__(state_dim, action_dim)

        self._hidden0 = torch.nn.Linear(state_dim, 400)
        self._hidden1 = torch.nn.Linear(400, 300)
        self._output = torch.nn.Linear(300, action_dim)

        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-3, 3e-3)

    def forward(self, state):
        x = torch.nn.functional.relu(self._hidden0(state))
        x = torch.nn.functional.relu(self._hidden1(x))
        policy = torch.tanh(self._output(x))
        return policy


def test(env, agent):
    state0 = torch.tensor(env.reset(), dtype=torch.float32)
    done = False
    total_rewards = 0

    while not done:
        # env.render()
        action = agent.get_action(state0)
        next_state, reward, done, _ = env.step(action.detach().numpy())
        total_rewards += reward
        state0 = torch.tensor(next_state, dtype=torch.float32)
    # env.render()
    env.close()
    return total_rewards


def run():
    epochs = 100
    env = gym.make('MountainCarContinuous-v0')

    agent = DDPG(Actor, Critic, env.observation_space.shape[0], env.action_space.shape[0], 100000, 64, 1e-4, 1e-3, 0.99, 1e-3, 1e-2)
    # exploration = GaussianExploration(0.2)
    exploration = OUExploration(env.action_space.shape[0], 0.2)

    for e in range(epochs):
        state0 = torch.tensor(env.reset(), dtype=torch.float32)
        done = False

        while not done:
            action0 = exploration.explore(agent.get_action(state0))
            # env.render()
            next_state, reward, done, _ = env.step(action0.detach().numpy())
            state1 = torch.tensor(next_state, dtype=torch.float32)
            agent.train(state0, action0, state1, reward, done)
            state0 = state1

        total_rewards = test(env, agent)
        exploration.reset()
        print('Episode ' + str(e) + ' reward ' + str(total_rewards))

    env.close()
