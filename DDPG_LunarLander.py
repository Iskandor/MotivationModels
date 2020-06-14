import gym
import numpy
import torch
from etaprogress.progress import ProgressBar

from algorithms.DDPG import DDPG, DDPGCritic, DDPGActor
from exploration.ContinuousExploration import GaussianExploration
from utils.Logger import Logger


class Critic(DDPGCritic):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__(state_dim, action_dim)

        self._hidden0 = torch.nn.Linear(state_dim, 160)
        self._hidden1 = torch.nn.Linear(160 + action_dim, 120)
        self._output = torch.nn.Linear(120, 1)

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

        self._hidden0 = torch.nn.Linear(state_dim, 160)
        self._hidden1 = torch.nn.Linear(160, 120)
        self._output = torch.nn.Linear(120, action_dim)

        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-1, 3e-1)

    def forward(self, state):
        x = torch.nn.functional.relu(self._hidden0(state))
        x = torch.nn.functional.relu(self._hidden1(x))
        policy = torch.tanh(self._output(x))
        return policy


def test(env, agent, render=False):
    state0 = torch.tensor(env.reset(), dtype=torch.float32)
    done = False
    total_rewards = 0

    while not done:
        if render:
            env.render()
        action = agent.get_action(state0)
        next_state, reward, done, _ = env.step(action.detach().numpy())
        total_rewards += reward
        state0 = torch.tensor(next_state, dtype=torch.float32)
    if render:
        env.render()
    return total_rewards


def run_baseline(trials, episodes, batch_size, memory_size):
    env = gym.make('LunarLanderContinuous-v2')
    log = Logger()
    log.disable()

    for i in range(trials):
        rewards = numpy.zeros(episodes)
        log.start()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = DDPG(Actor, Critic, state_dim, action_dim, memory_size, batch_size, 1e-4, 2e-4, 0.99, 1e-3)
        exploration = GaussianExploration(0.2)
        # exploration = OUExploration(env.action_space.shape[0], 0.2, mu=0.4)
        bar = ProgressBar(episodes, max_width=40)

        for e in range(episodes):
            state0 = torch.tensor(env.reset(), dtype=torch.float32)
            done = False
            train_reward = 0
            bar.numerator = e

            while not done:
                #env.render()
                action0 = exploration.explore(agent.get_action(state0))
                next_state, reward, done, _ = env.step(action0.detach().numpy())
                train_reward += reward
                state1 = torch.tensor(next_state, dtype=torch.float32)
                agent.train(state0, action0, state1, reward, done)
                state0 = state1

            test_reward = test(env, agent)
            rewards[e] = test_reward
            # visualize_policy(agent, i * epochs + e)
            # exploration.reset()
            print('Episode ' + str(e) + ' train reward ' + str(train_reward) + ' test reward ' + str(test_reward))
            print(bar)

            # log.log(str(test_reward) + '\n')
        log.close()

        # test(env, agent, True)
        # plot_graph(rewards, 'DDPG baseline trial ' + str(i), 'ddpg_baseline' + str(i))
        numpy.save('ddpg_baseline_' + str(i), rewards)

    env.close()
