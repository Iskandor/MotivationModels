import gym
import numpy
import torch
from etaprogress.progress import ProgressBar

from algorithms.DDPG import DDPG, DDPGCritic, DDPGActor
from exploration.ContinuousExploration import GaussianExploration
from motivation.ForwardModelMotivation import ForwardModel, ForwardModelMotivation
from motivation.MateLearnerMotivation import MetaLearnerModel, MetaLearnerMotivation
from utils.Logger import Logger


class Critic(DDPGCritic):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__(state_dim, action_dim)

        self._hidden0 = torch.nn.Linear(state_dim, 80)
        self._hidden1 = torch.nn.Linear(80 + action_dim, 60)
        self._output = torch.nn.Linear(60, 1)

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

        self._hidden0 = torch.nn.Linear(state_dim, 80)
        self._hidden1 = torch.nn.Linear(80, 60)
        self._output = torch.nn.Linear(60, action_dim)

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
        self._hidden0 = torch.nn.Linear(state_dim + action_dim, 80)
        self._hidden1 = torch.nn.Linear(80, 40)
        self._output = torch.nn.Linear(40, state_dim)

        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-1, 3e-1)

    def forward(self, state, action):
        x = torch.cat([state, action], state.ndim - 1)
        x = torch.nn.functional.relu(self._hidden0(x))
        x = torch.nn.functional.relu(self._hidden1(x))
        value = self._output(x)
        return value


class MetaLearnerNetwork(MetaLearnerModel):
    def __init__(self, state_dim, action_dim):
        super(MetaLearnerNetwork, self).__init__(state_dim, action_dim)
        self._hidden0 = torch.nn.Linear(state_dim + action_dim, 80)
        self._hidden1 = torch.nn.Linear(80, 40)
        self._output = torch.nn.Linear(40, 1)

        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-1, 3e-1)

    def forward(self, state, action):
        x = torch.cat([state, action], state.ndim - 1)
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
    return total_rewards


def run_baseline(args):
    env = gym.make('LunarLanderContinuous-v2')

    for i in range(args.trials):
        rewards = numpy.zeros(args.episodes)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = DDPG(Actor, Critic, state_dim, action_dim, args.memory_size, args.batch_size, 1e-4, 2e-4, 0.99, 1e-3)
        exploration = GaussianExploration(0.2)
        # exploration = OUExploration(env.action_space.shape[0], 0.2, mu=0.4)
        bar = ProgressBar(args.episodes, max_width=40)

        for e in range(args.episodes):
            state0 = torch.tensor(env.reset(), dtype=torch.float32)
            done = False
            train_reward = 0
            bar.numerator = e

            while not done:
                action0 = exploration.explore(agent.get_action(state0))
                next_state, reward, done, _ = env.step(action0.detach().numpy())
                train_reward += reward
                state1 = torch.tensor(next_state, dtype=torch.float32)
                agent.train(state0, action0, state1, reward, done)
                state0 = state1

            test_reward = test(env, agent)
            rewards[e] = test_reward
            print('Episode ' + str(e) + ' train reward ' + str(train_reward) + ' test reward ' + str(test_reward))
            print(bar)

        agent.save('./models/lunar_lander_baseline_' + str(i))
        numpy.save('ddpg_baseline_' + str(i), rewards)

    env.close()


def run_forward_model(args):
    env = gym.make('LunarLanderContinuous-v2')

    if args.load:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        motivation = ForwardModelMotivation(ForwardModelNetwork, state_dim, action_dim, 2e-4)
        agent = DDPG(Actor, Critic, state_dim, action_dim, args.memory_size, args.batch_size, 1e-4, 2e-4, 0.99, 1e-3, motivation_module=motivation)
        agent.load(args.load)

        test(env, agent, True)
    else:
        for i in range(args.trials):
            rewards = numpy.zeros(args.episodes)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            motivation = ForwardModelMotivation(ForwardModelNetwork, state_dim, action_dim, 2e-4)
            agent = DDPG(Actor, Critic, state_dim, action_dim, args.memory_size, args.batch_size, 1e-4, 2e-4, 0.99, 1e-3, motivation_module=motivation)
            exploration = GaussianExploration(0.2)
            # exploration = OUExploration(env.action_space.shape[0], 0.2, mu=0.4)
            bar = ProgressBar(args.episodes, max_width=40)

            for e in range(args.episodes):
                state0 = torch.tensor(env.reset(), dtype=torch.float32)
                done = False
                train_reward = 0
                bar.numerator = e

                while not done:
                    action0 = exploration.explore(agent.get_action(state0))
                    next_state, reward, done, _ = env.step(action0.detach().numpy())
                    train_reward += reward
                    state1 = torch.tensor(next_state, dtype=torch.float32)
                    agent.train(state0, action0, state1, reward, done)
                    motivation.train(state0, action0, state1)
                    state0 = state1

                test_reward = test(env, agent)
                rewards[e] = test_reward
                print('Episode ' + str(e) + ' train reward ' + str(train_reward) + ' test reward ' + str(test_reward))
                print(bar)

            agent.save('./models/lunar_lander_fm_' + str(i))
            numpy.save('ddpg_fm_' + str(i), rewards)

    env.close()


def run_metalearner_model(args):
    env = gym.make('LunarLanderContinuous-v2')

    if args.load:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        forward_model = ForwardModelMotivation(ForwardModelNetwork, state_dim, action_dim, 2e-4)
        motivation = MetaLearnerMotivation(MetaLearnerNetwork, forward_model, state_dim, action_dim, 2e-4)
        agent = DDPG(Actor, Critic, state_dim, action_dim, args.memory_size, args.batch_size, 1e-4, 2e-4, 0.99, 1e-3, motivation_module=motivation)
        agent.load(args.load)

        test(env, agent, True)
    else:
        for i in range(args.trials):
            rewards = numpy.zeros(args.episodes)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            forward_model = ForwardModelMotivation(ForwardModelNetwork, state_dim, action_dim, 2e-4)
            motivation = MetaLearnerMotivation(MetaLearnerNetwork, forward_model, state_dim, action_dim, 2e-4)

            agent = DDPG(Actor, Critic, state_dim, action_dim, args.memory_size, args.batch_size, 1e-4, 2e-4, 0.99, 1e-3, motivation_module=motivation)
            exploration = GaussianExploration(0.2)
            # exploration = OUExploration(env.action_space.shape[0], 0.2, mu=0.4)
            bar = ProgressBar(args.episodes, max_width=40)

            for e in range(args.episodes):
                state0 = torch.tensor(env.reset(), dtype=torch.float32)
                done = False
                train_reward = 0
                bar.numerator = e

                while not done:
                    action0 = exploration.explore(agent.get_action(state0))
                    next_state, reward, done, _ = env.step(action0.detach().numpy())
                    train_reward += reward
                    state1 = torch.tensor(next_state, dtype=torch.float32)
                    agent.train(state0, action0, state1, reward, done)
                    motivation.train(state0, action0, state1)
                    state0 = state1

                test_reward = test(env, agent)
                rewards[e] = test_reward
                print('Episode ' + str(e) + ' train reward ' + str(train_reward) + ' test reward ' + str(test_reward))
                print(bar)

            agent.save('./models/lunar_lander_su_' + str(i))
            numpy.save('ddpg_su_' + str(i), rewards)

    env.close()
