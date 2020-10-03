import gym
import torch
import numpy as np

from algorithms.A2C import A2C
from algorithms.Policy import Policy
from algorithms.TD import TD
from exploration.DiscreteExploration import DiscreteExploration


class ACNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ACNetwork, self).__init__()
        hidden = 128
        # limit = 0.001

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, int(hidden / 2)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden / 2), 1)
        )

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, int(hidden / 2)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden / 2), action_dim)
        )

    def forward(self, state):
        policy = torch.softmax(self.actor(state), dim=-1)
        value = self.critic(state)
        return policy, value


def run():
    epochs = 1000

    env = gym.make('CartPole-v0')

    network = ACNetwork(env.observation_space.shape[0], env.action_space.n)
    agent = A2C(network.actor, network.critic, 1e-3, 2e-3, 0.99)

    rewards = torch.zeros(100, dtype=torch.float32)
    reward_index = 0

    for e in range(epochs):
        state0 = torch.tensor(env.reset(), dtype=torch.float32)
        done = False
        total_reward = 0

        while not done:
            # env.render()
            action0 = agent.get_action(state0)
            next_state, reward, done, info = env.step(action0)
            total_reward += reward
            agent.train(reward, done)
            state1 = torch.tensor(next_state, dtype=torch.float32)
            state0 = state1

        rewards[reward_index] = total_reward
        reward_index += 1
        if reward_index == 100:
            reward_index = 0

        avg_reward = rewards.sum() / 100
        print('Episode ' + str(e) + ' reward ' + str(avg_reward.item()))

    env.close()
