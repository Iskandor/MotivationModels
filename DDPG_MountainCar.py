import gym
import numpy
import torch

from exploration.ContinuousExploration import OUExploration
from algorithms.DDPG import DDPG, DDPGCritic, DDPGActor
from motivation.ForwardModelMotivation import ForwardModel, ForwardModelMotivation
from motivation.MateLearnerMotivation import MetaLearnerMotivation, MetaLearnerModel
from utils.Logger import Logger

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

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
        torch.nn.init.uniform_(self._output.weight, -3e-1, 3e-1)

    def forward(self, state):
        x = torch.nn.functional.relu(self._hidden0(state))
        x = torch.nn.functional.relu(self._hidden1(x))
        policy = torch.tanh(self._output(x))
        return policy


class ForwardModelNetwork(ForwardModel):
    def __init__(self, state_dim, action_dim):
        super(ForwardModelNetwork, self).__init__(state_dim, action_dim)
        self._hidden0 = torch.nn.Linear(state_dim + action_dim, 100)
        self._hidden1 = torch.nn.Linear(100, 50)
        self._output = torch.nn.Linear(50, state_dim)

        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.cat([state, action])
        x = torch.nn.functional.relu(self._hidden0(x))
        x = torch.nn.functional.relu(self._hidden1(x))
        value = self._output(x)
        return value


class MetaLearnerNetwork(MetaLearnerModel):
    def __init__(self, state_dim, action_dim):
        super(MetaLearnerNetwork, self).__init__(state_dim, action_dim)
        self._hidden0 = torch.nn.Linear(state_dim + action_dim, 40)
        self._hidden1 = torch.nn.Linear(40, 30)
        self._output = torch.nn.Linear(30, 1)

        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.cat([state, action])
        x = torch.nn.functional.relu(self._hidden0(x))
        x = torch.nn.functional.relu(self._hidden1(x))
        value = self._output(x)
        return value


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
    env.close()
    return total_rewards


def test_forward_model(env, motivation):
    state0 = torch.tensor([numpy.random.rand(2)], dtype=torch.float32)
    state0[0] = state0[0] * 1.8 - 1.2
    state0[1] = state0[1] * 0.14 - 0.07

    env.state = state0.numpy()
    action = numpy.random.rand(1) * 2 - 1
    next_state, _, _, _ = env.step(action)
    state1 = torch.tensor(next_state, dtype=torch.float32)

    motivation.train()

def visualize_policy(agent, index=0):
    X, Y = numpy.meshgrid(numpy.linspace(-1.2, 0.6, 20), numpy.linspace(-0.07, 0.07, 20))
    input = torch.stack([torch.tensor(X, dtype=torch.float32).reshape(20 * 20), torch.tensor(Y, dtype=torch.float32).reshape(20 * 20)]).transpose(1, 0)
    Z = agent.get_action(input).reshape(20,20).detach().numpy()

    plt.pcolor(X, Y, Z, cmap='RdGy', vmin=-1, vmax=1)
    plt.colorbar()
    plt.savefig(str(index) + '.png')
    plt.clf()

def visualize_forward_model(env, agent, motivation, index=0):
    X, Y = numpy.meshgrid(numpy.linspace(-1.2, 0.6, 20), numpy.linspace(-0.07, 0.07, 20))
    input = torch.stack([torch.tensor(X, dtype=torch.float32).reshape(20 * 20), torch.tensor(Y, dtype=torch.float32).reshape(20 * 20)]).transpose(1, 0)
    Z = agent.get_action(input).reshape(20,20).detach().numpy()
    R = numpy.zeros((400))

    for i in range(input.shape[0]):
        env.state = input[i].numpy()
        action = agent.get_action(input[i])
        next_state, _, _, _ = env.step(action.detach().numpy())
        state1 = torch.tensor(next_state, dtype=torch.float32)
        R[i] = motivation.reward(input[i], action, state1)
    R = R.reshape((20,20))

    plt.figure(figsize=(10.00, 4.80))
    plt.subplot(1,2,1)
    plt.pcolor(X, Y, Z, cmap='RdGy', vmin=-1, vmax=1)
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.pcolor(X, Y, R, cmap='Greys', vmin=0, vmax=1)
    plt.colorbar()
    plt.savefig(str(index) + '.png')
    #plt.show()
    plt.close()

def visualize_metalearner_mode(env, agent, motivation, index=0):
    X, Y = numpy.meshgrid(numpy.linspace(-1.2, 0.6, 20), numpy.linspace(-0.07, 0.07, 20))
    input = torch.stack([torch.tensor(X, dtype=torch.float32).reshape(20 * 20), torch.tensor(Y, dtype=torch.float32).reshape(20 * 20)]).transpose(1, 0)
    Z = agent.get_action(input).reshape(20,20).detach().numpy()
    R = numpy.zeros((400))

    for i in range(input.shape[0]):
        env.state = input[i].numpy()
        action = agent.get_action(input[i])
        next_state, _, _, _ = env.step(action.detach().numpy())
        state1 = torch.tensor(next_state, dtype=torch.float32)
        R[i] = motivation.reward(input[i], action, state1)
    R = R.reshape((20,20))

    plt.figure(figsize=(10.00, 4.80))
    plt.subplot(1,2,1)
    plt.pcolor(X, Y, Z, cmap='RdGy', vmin=-1, vmax=1)
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.pcolor(X, Y, R, cmap='Greys', vmin=0, vmax=1)
    plt.colorbar()
    plt.savefig(str(index) + '.png')
    #plt.show()
    plt.close()

def run_baseline():
    epochs = 500
    env = gym.make('MountainCarContinuous-v0')
    log = Logger()
    log.enable()

    for i in range(7):
        log.start()
        agent = DDPG(Actor, Critic, env.observation_space.shape[0], env.action_space.shape[0], 10000, 64, 1e-4, 2e-4, 0.99, 1e-3)
        # exploration = GaussianExploration(0.2)
        exploration = OUExploration(env.action_space.shape[0], 0.2, mu=0.4)

        for e in range(epochs):
            state0 = torch.tensor(env.reset(), dtype=torch.float32)
            done = False
            train_reward = 0

            while not done:
                action0 = exploration.explore(agent.get_action(state0))
                # env.render()
                next_state, reward, done, _ = env.step(action0.detach().numpy())
                train_reward += reward
                state1 = torch.tensor(next_state, dtype=torch.float32)
                # agent.enable_gpu()
                agent.train(state0, action0, state1, reward, done)
                # agent.disable_gpu()
                state0 = state1
                # print(reward)

            test_reward = test(env, agent)
            #visualize_policy(agent, i * epochs + e)
            exploration.reset()
            print('Episode ' + str(e) + ' train reward ' + str(train_reward) + ' test reward ' + str(test_reward))
            log.log(str(test_reward) + '\n')
        log.close()

        test(env, agent, True)

    env.close()


def run_forward_model():
    epochs = 500
    env = gym.make('MountainCarContinuous-v0')
    log = Logger()
    log.enable()

    for i in range(7):
        log.start()
        agent = DDPG(Actor, Critic, env.observation_space.shape[0], env.action_space.shape[0], 10000, 64, 1e-4, 2e-4, 0.99, 1e-3)
        motivation = ForwardModelMotivation(ForwardModelNetwork, env.observation_space.shape[0], env.action_space.shape[0], 2e-4)
        # exploration = GaussianExploration(0.2)
        exploration = OUExploration(env.action_space.shape[0], 0.2, mu=0.4)

        for e in range(epochs):
            state0 = torch.tensor(env.reset(), dtype=torch.float32)
            done = False
            train_reward = 0

            while not done:
                action0 = exploration.explore(agent.get_action(state0))
                # env.render()
                next_state, ext_reward, done, _ = env.step(action0.detach().numpy())
                train_reward += ext_reward
                state1 = torch.tensor(next_state, dtype=torch.float32)
                int_reward = motivation.reward(state0, action0, state1, 1)
                train_reward += int_reward
                # agent.enable_gpu()
                agent.train(state0, action0, state1, ext_reward + int_reward, done)
                motivation.train(state0, action0, state1)
                # agent.disable_gpu()
                state0 = state1
                # print(ext_reward + int_reward)

            test_reward = test(env, agent)
            exploration.reset()
            #visualize_forward_model(env, agent, motivation, i * epochs + e)
            print('Episode ' + str(e) + ' train reward ' + str(train_reward) + ' test reward ' + str(test_reward))
            log.log(str(test_reward) + '\n')
        log.close()

        test(env, agent, True)

    env.close()


def run_metalearner_model():
    epochs = 500
    env = gym.make('MountainCarContinuous-v0')
    log = Logger()
    log.enable()

    for i in range(7):
        log.start()
        agent = DDPG(Actor, Critic, env.observation_space.shape[0], env.action_space.shape[0], 10000, 64, 1e-4, 2e-4, 0.99, 1e-3)
        forward_model = ForwardModelMotivation(ForwardModelNetwork, env.observation_space.shape[0], env.action_space.shape[0], 2e-4)
        motivation = MetaLearnerMotivation(MetaLearnerNetwork, forward_model, env.observation_space.shape[0], env.action_space.shape[0], 2e-4)

        # exploration = GaussianExploration(0.2)
        exploration = OUExploration(env.action_space.shape[0], 0.2, mu=0.4)

        for e in range(epochs):
            state0 = torch.tensor(env.reset(), dtype=torch.float32)
            done = False
            train_reward = 0

            while not done:
                action0 = exploration.explore(agent.get_action(state0))
                # env.render()
                next_state, ext_reward, done, _ = env.step(action0.detach().numpy())
                train_reward += ext_reward
                state1 = torch.tensor(next_state, dtype=torch.float32)
                int_reward = motivation.reward(state0, action0, state1)
                train_reward += int_reward
                # agent.enable_gpu()
                agent.train(state0, action0, state1, ext_reward + int_reward, done)
                motivation.train(state0, action0, state1)
                # agent.disable_gpu()
                state0 = state1
                # print(ext_reward + int_reward)

            test_reward = test(env, agent)
            exploration.reset()
            #visualize_metalearner_mode(env, agent, motivation, i * epochs + e)
            print('Episode ' + str(e) + ' train reward ' + str(train_reward) + ' test reward ' + str(test_reward))
            log.log(str(test_reward) + '\n')
        log.close()

        test(env, agent, True)

    env.close()
