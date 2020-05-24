import gym
import torch
import matplotlib.pyplot as plt

from algorithms.DDPG import DDPG, DDPGCritic, DDPGActor
from exploration.ContinuousExploration import GaussianExploration
from utils.Logger import Logger


class Critic(DDPGCritic):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__(state_dim, action_dim)

        self._hidden0 = torch.nn.Linear(state_dim, 256)
        self._hidden0_f = torch.nn.ReLU()
        self._hidden1 = torch.nn.Linear(256 + action_dim, 256)
        self._hidden1_f = torch.nn.ReLU()
        self._hidden2 = torch.nn.Linear(256, 256)
        self._hidden2_f = torch.nn.ReLU()
        self._output = torch.nn.Linear(256, action_dim)

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

        self._hidden0 = torch.nn.Linear(state_dim, 256)
        self._hidden0_f = torch.nn.ReLU()
        self._hidden1 = torch.nn.Linear(256, 256)
        self._hidden1_f = torch.nn.ReLU()
        self._hidden2 = torch.nn.Linear(256, 256)
        self._hidden2_f = torch.nn.ReLU()
        self._output = torch.nn.Linear(256, action_dim)
        self._output_f = torch.nn.Tanh()

        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.xavier_uniform_(self._hidden2.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-1, 3e-1)

    def forward(self, state):
        x = self._hidden0_f(self._hidden0(state))
        x = self._hidden1_f(self._hidden1(x))
        x = self._hidden2_f(self._hidden2(x))
        action = self._output_f(self._output(x))
        return action


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


def run_baseline(trials, episodes):
    env = gym.make('FetchReach-v1')
    log = Logger()
    log.disable()

    for i in range(trials):
        rewards = []
        log.start()
        state_dim = env.observation_space['achieved_goal'].shape[0] + env.observation_space['desired_goal'].shape[0]
        action_dim = env.action_space.shape[0]
        agent = DDPG(Actor, Critic, state_dim, action_dim, 1000000, 256, 1e-3, 1e-3, 0.99, 5e-3)
        exploration = GaussianExploration(0.2)
        # exploration = OUExploration(env.action_space.shape[0], 0.2, mu=0.4)

        for e in range(episodes):
            state0 = encode_state(env.reset())
            done = False
            train_reward = 0

            while not done:
                #env.render()
                action0 = exploration.explore(agent.get_action(state0))
                next_state, reward, done, _ = env.step(action0.detach().numpy())
                train_reward += reward
                state1 = encode_state(next_state)
                agent.train(state0, action0, state1, reward, done)
                state0 = state1

            test_reward = test(env, agent)
            rewards.append(test_reward)
            # visualize_policy(agent, i * epochs + e)
            # exploration.reset()
            print('Episode ' + str(e) + ' train reward ' + str(train_reward) + ' test reward ' + str(test_reward))
            # log.log(str(test_reward) + '\n')
        log.close()

        # test(env, agent, True)
        plot_graph(rewards, 'DDPG baseline trial ' + str(i), 'ddpg_baseline' + str(i))

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
