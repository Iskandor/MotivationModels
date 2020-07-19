import random
import time

import gym
import numpy
import torch
from etaprogress.progress import ProgressBar
from gym import wrappers
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from algorithms.DDPG import DDPG, DDPGCritic, DDPGActor
from exploration.ContinuousExploration import GaussianExploration
from motivation.ForwardModelMotivation import ForwardModel, ForwardModelMotivation
from motivation.MateLearnerMotivation import MetaLearnerModel, MetaLearnerMotivation
from sklearn.cluster import KMeans


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


class ForwardModelNetwork(ForwardModel):
    def __init__(self, state_dim, action_dim):
        super(ForwardModelNetwork, self).__init__(state_dim, action_dim)
        self._hidden0 = torch.nn.Linear(state_dim + action_dim, 200)
        self._hidden1 = torch.nn.Linear(200, 120)
        self._output = torch.nn.Linear(120, state_dim)

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
        self._hidden0 = torch.nn.Linear(state_dim + action_dim, 200)
        self._hidden1 = torch.nn.Linear(200, 120)
        self._output = torch.nn.Linear(120, 1)

        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-1, 3e-1)

    def forward(self, state, action):
        x = torch.cat([state, action], state.ndim - 1)
        x = torch.nn.functional.relu(self._hidden0(x))
        x = torch.nn.functional.relu(self._hidden1(x))
        value = self._output(x)
        return value


def test(env, agent, metacritic=None, forward_model=None, render=False, video=False):
    state0 = torch.tensor(env.reset(), dtype=torch.float32)
    done = False
    ext_rewards = 0
    int_rewards = 0
    steps = 0
    fm_error = []
    mc_error = []

    video_recorder = None
    if video:
        video_path = './videos/lunar_lander_baseline.mp4'
        video_recorder = VideoRecorder(env, video_path, enabled=video_path is not None)

    while not done:
        steps += 1
        if render:
            env.render()
            if video:
                env.unwrapped.render()
                video_recorder.capture_frame()

        action = agent.get_action(state0)
        next_state, reward, done, _ = env.step(action.detach().numpy())
        next_state = torch.tensor(next_state, dtype=torch.float32)

        ext_rewards += reward
        if metacritic is not None:
            int_rewards += metacritic.reward(state0, action, next_state).item()
            fm_error.append(forward_model.error(state0, action, next_state).item())
            mc_error.append(metacritic.error(state0, action).item())
        elif forward_model is not None:
            int_rewards += forward_model.reward(state0, action, next_state).item()
            fm_error.append(forward_model.error(state0, action, next_state).item())

        state0 = next_state
    if render:
        env.render()
    if video:
        video_recorder.close()
        video_recorder.enabled = False

    return ext_rewards, int_rewards, steps, fm_error, mc_error


def run_baseline(args):
    env = gym.make('LunarLanderContinuous-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    states = None
    if args.generate_states:
        states = []
    if args.collect_stats:
        states = torch.tensor(numpy.load('./lunar_lander_states.npy'), dtype=torch.float32)

    if args.load:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = DDPG(Actor, Critic, state_dim, action_dim, args.memory_size, args.batch_size, 1e-4, 2e-4, 0.99, 1e-3)
        agent.load(args.load)

        for i in range(5):
            test(env, agent, True, False)
    else:
        action_list = []
        value_list = []

        for i in range(args.trials):
            test_ext_rewards = numpy.zeros(args.episodes)
            test_int_rewards = numpy.zeros(args.episodes)
            agent = DDPG(Actor, Critic, state_dim, action_dim, args.memory_size, args.batch_size, 1e-4, 2e-4, 0.99, 1e-3, motivation_module=None)
            exploration = GaussianExploration(0.2)
            # exploration = OUExploration(env.action_space.shape[0], 0.2, mu=0.4)
            bar = ProgressBar(args.episodes, max_width=40)

            for e in range(args.episodes):
                if args.collect_stats:
                    actions, values = baseline_activations(agent, states)
                    action_list.append(actions)
                    value_list.append(values)

                state0 = torch.tensor(env.reset(), dtype=torch.float32)
                done = False
                train_ext_reward = 0
                bar.numerator = e
                train_steps = 0

                t0 = time.perf_counter()
                while not done:
                    train_steps += 1
                    if args.generate_states:
                        states.append(state0.numpy())
                    action0 = exploration.explore(agent.get_action(state0))
                    next_state, reward, done, _ = env.step(action0.numpy())
                    train_ext_reward += reward
                    # if not done and random.random() < 0.9:
                    #    reward = 0
                    state1 = torch.tensor(next_state, dtype=torch.float32)
                    agent.train(state0, action0, state1, reward, done)
                    # print("external reward {0:f}, internal reward {1:f}".format(reward, ri.item()))
                    state0 = state1
                t1 = time.perf_counter()
                print('Training ' + str(t1 - t0))
                t0 = time.perf_counter()
                test_ext_reward, _, test_steps, _, _ = test(env, agent, metacritic=None, forward_model=None)
                t1 = time.perf_counter()
                print('Testing ' + str(t1 - t0))
                test_ext_rewards[e] = test_ext_reward
                print(
                    'Episode {0:d} training [ext. reward {1:f} steps {2:d}] testing [ext. reward {3:f} steps {4:d}]'.format(
                        e, train_ext_reward, train_steps, test_ext_reward, test_steps))
                print(bar)

            agent.save('./models/lunar_lander_baseline_{0:d}'.format(i))
            numpy.save('ddpg_baseline_{0:d}_re'.format(i), test_ext_rewards)

            if args.generate_states:
                kmeans = KMeans(n_clusters=2000, random_state=0).fit(states)
                states = numpy.stack(kmeans.cluster_centers_)
                states[:, 6] = numpy.round(states[:, 6])
                states[:, 7] = numpy.round(states[:, 7])
                numpy.save('lunar_lander_states', states)

            if args.collect_stats:
                action_list = torch.stack(action_list)
                value_list = torch.stack(value_list)

                numpy.save('ddpg_baseline_' + str(i) + '_actions', action_list)
                numpy.save('ddpg_baseline_' + str(i) + '_values', value_list)

    env.close()


def run_forward_model(args):
    env = gym.make('LunarLanderContinuous-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    states = None
    if args.generate_states:
        states = []
    if args.collect_stats:
        states = torch.tensor(numpy.load('./lunar_lander_states.npy'), dtype=torch.float32)

    if args.load:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = DDPG(Actor, Critic, state_dim, action_dim, args.memory_size, args.batch_size, 1e-4, 2e-4, 0.99, 1e-3)
        agent.load(args.load)

        for i in range(5):
            test(env, agent, True, False)
    else:
        action_list = []
        value_list = []
        fm_error_list = []
        reward_list = []

        fm_train_errors = []

        for i in range(args.trials):
            test_ext_rewards = numpy.zeros(args.episodes)
            test_int_rewards = numpy.zeros(args.episodes)
            forward_model = ForwardModelMotivation(ForwardModelNetwork, state_dim, action_dim, 2e-4)

            agent = DDPG(Actor, Critic, state_dim, action_dim, args.memory_size, args.batch_size, 1e-4, 2e-4, 0.99, 1e-3, motivation_module=forward_model)
            exploration = GaussianExploration(0.2)
            # exploration = OUExploration(env.action_space.shape[0], 0.2, mu=0.4)
            bar = ProgressBar(args.episodes, max_width=40)

            for e in range(args.episodes):
                if args.collect_stats:
                    actions, values, fm_errors, rewards = fm_activations(env, agent, forward_model, states)
                    action_list.append(actions)
                    value_list.append(values)
                    fm_error_list.append(fm_errors)
                    reward_list.append(rewards)

                state0 = torch.tensor(env.reset(), dtype=torch.float32)
                done = False
                train_ext_reward = 0
                train_int_reward = 0
                bar.numerator = e
                train_steps = 0

                t0 = time.perf_counter()
                while not done:
                    train_steps += 1
                    if args.generate_states:
                        states.append(state0.numpy())
                    action0 = exploration.explore(agent.get_action(state0))
                    next_state, reward, done, _ = env.step(action0.numpy())
                    train_ext_reward += reward
                    # if not done and random.random() < 0.9:
                    #    reward = 0
                    state1 = torch.tensor(next_state, dtype=torch.float32)
                    agent.train(state0, action0, state1, reward, done)
                    forward_model.train(state0, action0, state1)
                    train_int_reward += forward_model.reward(state0, action0, state1).item()
                    # print("external reward {0:f}, internal reward {1:f}".format(reward, ri.item()))
                    state0 = state1
                t1 = time.perf_counter()
                print('Training ' + str(t1 - t0))
                t0 = time.perf_counter()
                test_ext_reward, test_int_reward, test_steps, fm_error, _ = test(env, agent, metacritic=None, forward_model=forward_model)
                fm_train_errors.append(fm_error)
                t1 = time.perf_counter()
                print('Testing ' + str(t1 - t0))
                test_ext_rewards[e] = test_ext_reward
                test_int_rewards[e] = test_int_reward
                print(
                    'Episode {0:d} training [ext. reward {1:f} int. reward {2:f} steps {3:d}] testing [ext. reward {4:f} int. reward {5:f} steps {6:d}]'.format(
                        e, train_ext_reward, train_int_reward, train_steps, test_ext_reward, test_int_reward, test_steps))
                print(bar)

            agent.save('./models/lunar_lander_fm_{0:d}'.format(i))
            numpy.save('ddpg_fm_{0:d}_re'.format(i), test_ext_rewards)
            numpy.save('ddpg_fm_{0:d}_ri'.format(i), test_int_rewards)

            if args.generate_states:
                kmeans = KMeans(n_clusters=2000, random_state=0).fit(states)
                states = numpy.stack(kmeans.cluster_centers_)
                states[:, 6] = numpy.round(states[:, 6])
                states[:, 7] = numpy.round(states[:, 7])
                numpy.save('lunar_lander_states', states)

            fm_train_errors = [item for sublist in fm_train_errors for item in sublist]
            fm_train_errors = numpy.stack(fm_train_errors)
            numpy.save('ddpg_fm_{0:d}_fme'.format(i), fm_train_errors)

            if args.collect_stats:
                action_list = torch.stack(action_list)
                value_list = torch.stack(value_list)
                fm_error_list = torch.stack(fm_error_list)
                reward_list = torch.stack(reward_list)

                numpy.save('ddpg_fm_' + str(i) + '_actions', action_list)
                numpy.save('ddpg_fm_' + str(i) + '_values', value_list)
                numpy.save('ddpg_fm_' + str(i) + '_prediction_errors', fm_error_list)
                numpy.save('ddpg_fm_' + str(i) + '_rewards', reward_list)

    env.close()


def run_metalearner_model(args):
    env = gym.make('LunarLanderContinuous-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    states = None
    if args.generate_states:
        states = []
    if args.collect_stats:
        states = torch.tensor(numpy.load('./lunar_lander_states.npy'), dtype=torch.float32)

    if args.load:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = DDPG(Actor, Critic, state_dim, action_dim, args.memory_size, args.batch_size, 1e-4, 2e-4, 0.99, 1e-3)
        agent.load(args.load)

        for i in range(5):
            test(env, agent, True, False)
    else:
        action_list = []
        value_list = []
        fm_error_list = []
        mc_error_list = []
        reward_list = []

        fm_train_errors = []
        mc_train_errors = []

        for i in range(args.trials):
            test_ext_rewards = numpy.zeros(args.episodes)
            test_int_rewards = numpy.zeros(args.episodes)
            forward_model = ForwardModelMotivation(ForwardModelNetwork, state_dim, action_dim, 2e-4)
            metacritic = MetaLearnerMotivation(MetaLearnerNetwork, forward_model, state_dim, action_dim, 2e-3, variant='A', eta=1)

            agent = DDPG(Actor, Critic, state_dim, action_dim, args.memory_size, args.batch_size, 1e-4, 2e-4, 0.99, 1e-3, motivation_module=metacritic)
            exploration = GaussianExploration(0.2)
            # exploration = OUExploration(env.action_space.shape[0], 0.2, mu=0.4)
            bar = ProgressBar(args.episodes, max_width=40)

            for e in range(args.episodes):
                if args.collect_stats:
                    actions, values, fm_errors, mc_errors, rewards = su_activations(env, agent, forward_model, metacritic, states)
                    action_list.append(actions)
                    value_list.append(values)
                    fm_error_list.append(fm_errors)
                    mc_error_list.append(mc_errors)
                    reward_list.append(rewards)

                state0 = torch.tensor(env.reset(), dtype=torch.float32)
                done = False
                train_ext_reward = 0
                train_int_reward = 0
                bar.numerator = e
                train_steps = 0

                t0 = time.perf_counter()
                while not done:
                    train_steps += 1
                    if args.generate_states:
                        states.append(state0.numpy())
                    action0 = exploration.explore(agent.get_action(state0))
                    next_state, reward, done, _ = env.step(action0.numpy())
                    train_ext_reward += reward
                    # if not done and random.random() < 0.9:
                    #    reward = 0
                    state1 = torch.tensor(next_state, dtype=torch.float32)
                    agent.train(state0, action0, state1, reward, done)
                    metacritic.train(state0, action0, state1)
                    train_int_reward += metacritic.reward(state0, action0, state1).item()
                    # print("external reward {0:f}, internal reward {1:f}".format(reward, ri.item()))
                    state0 = state1
                t1 = time.perf_counter()
                print('Training ' + str(t1 - t0))
                t0 = time.perf_counter()
                test_ext_reward, test_int_reward, test_steps, fm_error, mc_error = test(env, agent, metacritic=metacritic, forward_model=forward_model)
                fm_train_errors.append(fm_error)
                mc_train_errors.append(mc_error)
                t1 = time.perf_counter()
                print('Testing ' + str(t1 - t0))
                test_ext_rewards[e] = test_ext_reward
                test_int_rewards[e] = test_int_reward
                print(
                    'Episode {0:d} training [ext. reward {1:f} int. reward {2:f} steps {3:d}] testing [ext. reward {4:f} int. reward {5:f} steps {6:d}]'.format(
                        e, train_ext_reward, train_int_reward, train_steps, test_ext_reward, test_int_reward, test_steps))
                print(bar)

            agent.save('./models/lunar_lander_su_{0:d}'.format(i))
            numpy.save('ddpg_su_{0:d}_re'.format(i), test_ext_rewards)
            numpy.save('ddpg_su_{0:d}_ri'.format(i), test_int_rewards)

            if args.generate_states:
                kmeans = KMeans(n_clusters=2000, random_state=0).fit(states)
                states = numpy.stack(kmeans.cluster_centers_)
                states[:, 6] = numpy.round(states[:, 6])
                states[:, 7] = numpy.round(states[:, 7])
                numpy.save('lunar_lander_states', states)

            fm_train_errors = [item for sublist in fm_train_errors for item in sublist]
            fm_train_errors = numpy.stack(fm_train_errors)
            mc_train_errors = [item for sublist in mc_train_errors for item in sublist]
            mc_train_errors = numpy.stack(mc_train_errors)
            numpy.save('ddpg_su_{0:d}_fme'.format(i), fm_train_errors)
            numpy.save('ddpg_su_{0:d}_mce'.format(i), mc_train_errors)

            if args.collect_stats:
                action_list = torch.stack(action_list)
                value_list = torch.stack(value_list)
                fm_error_list = torch.stack(fm_error_list)
                mc_error_list = torch.stack(mc_error_list)
                reward_list = torch.stack(reward_list)

                numpy.save('ddpg_su_' + str(i) + '_actions', action_list)
                numpy.save('ddpg_su_' + str(i) + '_values', value_list)
                numpy.save('ddpg_su_' + str(i) + '_prediction_errors', fm_error_list)
                numpy.save('ddpg_su_' + str(i) + '_error_estimations', mc_error_list)
                numpy.save('ddpg_su_' + str(i) + '_rewards', reward_list)

    env.close()


def generate_states(n, state_dim):
    limits = torch.tensor([1, 2.5, 3.5, 3, 13, 7, 0, 0])
    states = (torch.rand(n, state_dim) * 2 - 1) * limits
    return states


def baseline_activations(agent, states):
    actions = agent.get_action(states)
    values = agent.get_value(states, actions)
    return actions, values


def fm_activations(env, agent, forward_model, states):
    actions, values = baseline_activations(agent, states)
    next_states = []
    env.reset()
    for i in range(states.shape[0]):
        env.set_state(states[i].numpy())
        next_state, _, _, _ = env.step(actions[i].numpy())
        next_states.append(torch.tensor(next_state))
    next_states = torch.stack(next_states)
    errors = forward_model.error(states, actions, next_states)
    rewards = forward_model.reward(states, actions, next_states)

    return actions, values, errors, rewards


def su_activations(env, agent, forward_model, metacritic, states):
    actions, values = baseline_activations(agent, states)
    next_states = []
    env.reset()
    for i in range(states.shape[0]):
        env.set_state(states[i].numpy())
        next_state, _, _, _ = env.step(actions[i].numpy())
        next_states.append(torch.tensor(next_state))
    next_states = torch.stack(next_states)
    fm_errors = forward_model.error(states, actions, next_states)
    mc_errors = metacritic.error(states, actions)
    rewards = metacritic.reward(states, actions, next_states)

    return actions, values, fm_errors, mc_errors, rewards
