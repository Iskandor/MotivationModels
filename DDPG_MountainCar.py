import gym
import torch

from exploration.ContinuousExploration import OUExploration
from algorithms.DDPG import DDPG, DDPGCritic, DDPGActor
from motivation.ForwardModelMotivation import ForwardModel, ForwardModelMotivation
from motivation.MateLearnerMotivation import MetaLearnerMotivation, MetaLearnerModel
from utils.Logger import Logger


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
        self._hidden0 = torch.nn.Linear(state_dim + action_dim, 40)
        self._hidden1 = torch.nn.Linear(40, 30)
        self._output = torch.nn.Linear(30, state_dim)

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


def run_baseline():
    epochs = 100
    env = gym.make('MountainCarContinuous-v0')
    log = Logger()
    log.enable()

    for _ in range(1):
        log.start()
        agent = DDPG(Actor, Critic, env.observation_space.shape[0], env.action_space.shape[0], 10000, 64, 1e-4, 1e-3, 0.99, 1e-3)
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
            exploration.reset()
            print('Episode ' + str(e) + ' train reward ' + str(train_reward) + ' test reward ' + str(test_reward))
            log.log(str(test_reward) + '\n')
        log.close()

        test(env, agent, True)

    env.close()


def run_forward_model():
    epochs = 100
    env = gym.make('MountainCarContinuous-v0')
    log = Logger()
    log.enable()

    for _ in range(1):
        log.start()
        agent = DDPG(Actor, Critic, env.observation_space.shape[0], env.action_space.shape[0], 10000, 64, 1e-4, 1e-3, 0.99, 1e-3)
        motivation = ForwardModelMotivation(ForwardModelNetwork, env.observation_space.shape[0], env.action_space.shape[0], 1e-3)
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
                # agent.disable_gpu()
                state0 = state1
                # print(ext_reward + int_reward)

            test_reward = test(env, agent)
            exploration.reset()
            print('Episode ' + str(e) + ' train reward ' + str(train_reward) + ' test reward ' + str(test_reward))
            log.log(str(test_reward) + '\n')
        log.close()

        test(env, agent, True)

    env.close()


def run_metalearner_model():
    epochs = 100
    env = gym.make('MountainCarContinuous-v0')
    log = Logger()
    log.enable()

    for _ in range(1):
        log.start()
        agent = DDPG(Actor, Critic, env.observation_space.shape[0], env.action_space.shape[0], 10000, 64, 1e-4, 1e-3, 0.99, 1e-3)
        forward_model = ForwardModelMotivation(ForwardModelNetwork, env.observation_space.shape[0], env.action_space.shape[0], 1e-3)
        motivation = MetaLearnerMotivation(MetaLearnerNetwork, forward_model, env.observation_space.shape[0], env.action_space.shape[0], 1e-3)

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
                # agent.disable_gpu()
                state0 = state1
                # print(ext_reward + int_reward)

            test_reward = test(env, agent)
            exploration.reset()
            print('Episode ' + str(e) + ' train reward ' + str(train_reward) + ' test reward ' + str(test_reward))
            log.log(str(test_reward) + '\n')
        log.close()

        test(env, agent, True)

    env.close()
