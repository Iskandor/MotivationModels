import gym
import numpy
import torch

from exploration.ContinuousExploration import OUExploration
from algorithms.DDPG import DDPG, DDPGCritic, DDPGActor
from motivation.ForwardModelMotivation import ForwardModel, ForwardModelMotivation
from motivation.MateLearnerMotivation import MetaLearnerMotivation, MetaLearnerModel
from utils.Logger import Logger

class Critic(DDPGCritic):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__(state_dim, action_dim)

        self._hidden0 = torch.nn.Linear(state_dim, 40)
        self._hidden1 = torch.nn.Linear(40 + action_dim, 30)
        self._output = torch.nn.Linear(30, 1)

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

        self._hidden0 = torch.nn.Linear(state_dim, 40)
        self._hidden1 = torch.nn.Linear(40, 30)
        self._output = torch.nn.Linear(30, action_dim)

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


class MetaLearnerNetwork(MetaLearnerModel):
    def __init__(self, state_dim, action_dim):
        super(MetaLearnerNetwork, self).__init__(state_dim, action_dim)
        self._hidden0 = torch.nn.Linear(state_dim + action_dim, 50)
        self._hidden1 = torch.nn.Linear(50, 30)
        self._output = torch.nn.Linear(30, 1)

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

def run_baseline(trials, episodes):
    env = gym.make('MountainCarContinuous-v0')
    log = Logger()
    log.enable()

    for i in range(trials):
        rewards = []
        log.start()
        agent = DDPG(Actor, Critic, env.observation_space.shape[0], env.action_space.shape[0], 10000, 64, 1e-4, 2e-4, 0.99, 1e-3)
        # exploration = GaussianExploration(0.2)
        exploration = OUExploration(env.action_space.shape[0], 0.2, mu=0.4)

        for e in range(episodes):
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
            rewards.append(test_reward)
            # visualize_policy(agent, i * epochs + e)
            exploration.reset()
            print('Episode ' + str(e) + ' train reward ' + str(train_reward) + ' test reward ' + str(test_reward))
            log.log(str(test_reward) + '\n')
        log.close()

        #test(env, agent, True)

    env.close()


def run_forward_model(trials, episodes):
    env = gym.make('MountainCarContinuous-v0')
    log = Logger()
    log.enable()

    for i in range(trials):
        rewards = []
        log.start()
        motivation = ForwardModelMotivation(ForwardModelNetwork, env.observation_space.shape[0], env.action_space.shape[0], 2e-4)
        agent = DDPG(Actor, Critic, env.observation_space.shape[0], env.action_space.shape[0], 10000, 64, 1e-4, 2e-4, 0.99, 1e-3, motivation_module=motivation)
        # exploration = GaussianExploration(0.2)
        exploration = OUExploration(env.action_space.shape[0], 0.2, mu=0.4)

        for e in range(episodes):
            state0 = torch.tensor(env.reset(), dtype=torch.float32)
            done = False
            train_reward_ext = 0
            train_reward_int = 0

            while not done:
                action0 = exploration.explore(agent.get_action(state0))
                # env.render()
                next_state, ext_reward, done, _ = env.step(action0.detach().numpy())
                train_reward_ext += ext_reward
                state1 = torch.tensor(next_state, dtype=torch.float32)
                int_reward = motivation.reward(state0, action0, state1, 1)
                train_reward_int += int_reward
                # agent.enable_gpu()
                agent.train(state0, action0, state1, ext_reward, done)
                motivation.train(state0, action0, state1)
                # agent.disable_gpu()
                state0 = state1
                # print(ext_reward + int_reward)

            test_reward = test(env, agent)
            rewards.append(test_reward)
            exploration.reset()
            # visualize_forward_model(env, agent, motivation, i * epochs + e)
            print('Episode ' + str(e) + ' train ext. reward ' + str(train_reward_ext) + ' int. reward ' + str(train_reward_int) + ' test reward ' + str(test_reward))
            log.log(str(test_reward) + '\n')
        log.close()

        # test(env, agent, True)

    env.close()


def run_metalearner_model(trials, episodes):
    env = gym.make('MountainCarContinuous-v0')
    log = Logger()
    log.enable()

    for i in range(trials):
        rewards = []
        images = []
        log.start('ddpg_su_model' + str(i))
        forward_model = ForwardModelMotivation(ForwardModelNetwork, env.observation_space.shape[0], env.action_space.shape[0], 2e-4)
        motivation = MetaLearnerMotivation(MetaLearnerNetwork, forward_model, env.observation_space.shape[0], env.action_space.shape[0], 2e-4)
        agent = DDPG(Actor, Critic, env.observation_space.shape[0], env.action_space.shape[0], 10000, 64, 1e-4, 2e-4, 0.99, 1e-3, motivation_module=motivation)

        # exploration = GaussianExploration(0.2)
        exploration = OUExploration(env.action_space.shape[0], 0.2, mu=0.4)

        # env.reset()
        # visualize_metalearner_model(env, agent, forward_model, motivation, 0)


        for e in range(episodes):
            state0 = torch.tensor(env.reset(), dtype=torch.float32)
            done = False
            train_reward_ext = 0
            train_reward_int = 0

            while not done:
                action0 = exploration.explore(agent.get_action(state0))
                # env.render()
                next_state, ext_reward, done, _ = env.step(action0.detach().numpy())
                train_reward_ext += ext_reward
                state1 = torch.tensor(next_state, dtype=torch.float32)
                int_reward = motivation.reward('A', state0, action0, state1)
                train_reward_int += int_reward
                # agent.enable_gpu()
                agent.train(state0, action0, state1, ext_reward, done)
                motivation.train(state0, action0, state1)
                # agent.disable_gpu()
                state0 = state1
                # print(ext_reward + int_reward)

            test_reward = test(env, agent)
            rewards.append(test_reward)
            exploration.reset()
            print('Episode ' + str(e) + ' train ext. reward ' + str(train_reward_ext) + ' int. reward ' + str(train_reward_int) + ' test reward ' + str(test_reward))
            log.log(str(test_reward) + '\n')
        log.close()

    env.close()
