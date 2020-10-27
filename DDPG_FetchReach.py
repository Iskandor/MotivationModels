import gym
import numpy
import torch
import matplotlib.pyplot as plt

from algorithms.DDPG import DDPG, DDPGCritic, DDPGActor
from exploration.ContinuousExploration import GaussianExploration
from motivation.ForwardModelMotivation import ForwardModel, ForwardModelMotivation
from utils.Logger import Logger
from etaprogress.progress import ProgressBar


class Critic(DDPGCritic):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__(state_dim, action_dim)

        self._hidden0 = torch.nn.Linear(state_dim, 120)
        self._hidden0_f = torch.nn.ReLU()
        self._hidden1 = torch.nn.Linear(120 + action_dim, 80)
        self._hidden1_f = torch.nn.ReLU()
        self._hidden2 = torch.nn.Linear(80, 60)
        self._hidden2_f = torch.nn.ReLU()
        self._output = torch.nn.Linear(60, action_dim)

        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.xavier_uniform_(self._hidden2.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-3, 3e-3)

    def forward(self, state, action):
        x = self._hidden0_f(self._hidden0(state))
        x = torch.cat([x, action], 1)
        x = self._hidden1_f(self._hidden1(x))
        x = self._hidden2_f(self._hidden2(x))
        value = self._output(x)
        return value


class Actor(DDPGActor):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__(state_dim, action_dim)

        self._hidden0 = torch.nn.Linear(state_dim, 120)
        self._hidden0_f = torch.nn.ReLU()
        self._hidden1 = torch.nn.Linear(120, 80)
        self._hidden1_f = torch.nn.ReLU()
        self._output = torch.nn.Linear(80, action_dim)
        self._output_f = torch.nn.Tanh()

        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-3, 3e-3)

    def forward(self, state):
        x = self._hidden0_f(self._hidden0(state))
        x = self._hidden1_f(self._hidden1(x))
        action = self._output_f(self._output(x))
        return action

class ForwardModelNetwork(ForwardModel):
    def __init__(self, state_dim, action_dim):
        super(ForwardModelNetwork, self).__init__(state_dim, action_dim)
        self._hidden0 = torch.nn.Linear(state_dim + action_dim, 50)
        self._hidden1 = torch.nn.Linear(50, 30)
        self._output = torch.nn.Linear(30, state_dim)

        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-1, 3e-1)

    def forward(self, state, action):
        x = torch.cat([state, action], state.ndim - 1)
        x = torch.nn.functional.relu(self._hidden0(x))
        x = torch.nn.functional.relu(self._hidden1(x))
        value = self._output(x)
        return value


def encode_state(state):
    achieved_goal = torch.tensor(state['achieved_goal'], dtype=torch.float32)
    desired_goal = torch.tensor(state['desired_goal'], dtype=torch.float32)
    return torch.cat((achieved_goal, desired_goal))


def test(env, agent, render=False):
    state0 = encode_state(env.reset())
    done = False
    total_rewards = 0

    while not done:
        if render:
            env.render()
        action = agent.get_action(state0)
        next_state, reward, done, _ = env.step(action.detach().numpy())
        total_rewards += reward
        state0 = encode_state(next_state)
    if render:
        env.render()
    env.close()
    return total_rewards


def run_baseline(args):
    env = gym.make('FetchReach-v1')
    log = Logger()
    log.disable()

    for i in range(args.trials):
        rewards = numpy.zeros(args.episodes)
        log.start()
        state_dim = env.observation_space['achieved_goal'].shape[0] + env.observation_space['desired_goal'].shape[0]
        action_dim = env.action_space.shape[0]
        agent = DDPG(Actor, Critic, state_dim, action_dim, args.memory_size, args.batch_size, 1e-3, 1e-3, 0.99, 5e-3)
        exploration = GaussianExploration(0.2)
        # exploration = OUExploration(env.action_space.shape[0], 0.2, mu=0.4)
        bar = ProgressBar(args.episodes, max_width=40)

        for e in range(args.episodes):
            state0 = encode_state(env.reset())
            done = False
            train_reward = 0
            bar.numerator = e

            while not done:
                action0 = exploration.explore(agent.get_action(state0))
                next_state, reward, done, _ = env.step(action0.detach().numpy())
                train_reward += reward
                state1 = encode_state(next_state)
                agent.train(state0, action0, state1, reward, done)
                state0 = state1

            test_reward = test(env, agent)
            rewards[e] = test_reward
            print('Episode ' + str(e) + ' train reward ' + str(train_reward) + ' test reward ' + str(test_reward))
            print(bar)

        log.close()

        numpy.save('ddpg_baseline_' + str(i), rewards)

    env.close()


def run_forward_model(trials, episodes, batch_size, memory_size):
    env = gym.make('FetchReach-v1')
    log = Logger()
    log.disable()

    for i in range(trials):
        train_rewards = numpy.zeros(episodes)
        intrinsic_rewards = numpy.zeros(episodes)
        test_rewards = numpy.zeros(episodes)
        log.start()
        state_dim = env.observation_space['achieved_goal'].shape[0] + env.observation_space['desired_goal'].shape[0]
        action_dim = env.action_space.shape[0]
        motivation = ForwardModelMotivation(ForwardModelNetwork, env.observation_space['achieved_goal'].shape[0], action_dim, 2e-4)
        agent = DDPG(Actor, Critic, state_dim, action_dim, memory_size, batch_size, 1e-4, 2e-4, 0.99, 5e-3)
        exploration = GaussianExploration(0.2)
        #exploration = OUExploration(env.action_space.shape[0], 0.2, mu=0.0)
        bar = ProgressBar(episodes, max_width=40)

        for e in range(episodes):
            state0 = encode_state(env.reset())
            done = False
            train_reward = 0
            intrinsic_reward = 0
            bar.numerator = e

            while not done:
                env.render()
                action0 = exploration.explore(agent.get_action(state0))
                next_state, reward, done, _ = env.step(action0.detach().numpy())
                train_reward += reward
                state1 = encode_state(next_state)
                agent.train(state0, action0, state1, reward, done)
                motivation.train(state0[:3], action0, state1[:3])
                intrinsic_reward += motivation.reward(state0[:3], action0, state1[:3])
                state0 = state1

            test_reward = test(env, agent)
            train_rewards[e] = train_reward
            test_rewards[e] = test_reward
            intrinsic_rewards[e] = intrinsic_reward
            # visualize_policy(agent, i * epochs + e)
            #exploration.reset()
            print('Episode ' + str(e) + ' train reward ' + str(train_reward) + ' intrinsic reward ' + str(intrinsic_reward) + ' test reward ' + str(test_reward))
            print(bar)

            # log.log(str(test_reward) + '\n')
        log.close()

        # test(env, agent, True)
        # plot_graph(rewards, 'DDPG baseline trial ' + str(i), 'ddpg_baseline' + str(i))
        numpy.save('ddpg_fm_erte' + str(i), test_rewards)
        numpy.save('ddpg_fm_ertr' + str(i), train_rewards)
        numpy.save('ddpg_fm_irtr' + str(i), intrinsic_rewards)

    env.close()

def plot_graph(p_data, p_title, p_filename):
    colors = ['blue', 'red', 'green']

    fig, ax = plt.subplots(1)
    ax.set_title(p_title)
    ax.set_xlabel('episodes')
    ax.set_ylabel('reward')
    ax.grid()
    ax.plot(p_data, lw=2, label='mean reward', color=colors[0])
    plt.savefig(p_filename + '.png')
