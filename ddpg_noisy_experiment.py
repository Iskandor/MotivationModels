import time

import gym
import pybulletgym
import numpy
import torch
from etaprogress.progress import ProgressBar
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from sklearn.cluster import KMeans

from exploration.ContinuousExploration import GaussianExploration


class ExperimentNoisyDDPG:
    def __init__(self, env_name, env, config):
        self._env_name = env_name
        self._env = env
        self._config = config
        self._actor = None
        self._critic = None
        self._preprocess = None

    def add_preprocess(self, preprocess):
        self._preprocess = preprocess

    def test(self, env, agent, metacritic=None, forward_model=None, render=False, video=False):
        state0 = torch.tensor(env.reset(), dtype=torch.float32)
        done = False
        ext_rewards = 0
        int_rewards = 0
        steps = 0
        fm_error = []
        mc_error = []

        video_recorder = None
        if video:
            video_path = './videos/{0:s}_baseline.mp4'.format(self._env_name)
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

    def run_baseline(self, agent, trial):
        config = self._config
        step_limit = config.steps * 1e6
        steps = 0

        states = None
        if config.stats.generate_states:
            states = []
        if config.stats.collect_stats:
            states = torch.tensor(numpy.load('./{0:s}_states.npy'.format(self._env_name)), dtype=torch.float32)

        if config.load:
            agent.load(config.load)

            for i in range(5):
                self.test(self._env, agent, render=False, video=False)
        else:
            action_list = []
            value_list = []

            train_ext_rewards = []
            bar = ProgressBar(config.steps * 1e6, max_width=40)

            while steps < step_limit:
                if config.stats.collect_stats:
                    actions, values = self.baseline_activations(agent, states)
                    action_list.append(actions)
                    value_list.append(values)

                if self._preprocess is None:
                    state0 = torch.tensor(self._env.reset(), dtype=torch.float32)
                else:
                    state0 = self._preprocess(self._env.reset())

                done = False
                train_ext_reward = 0
                bar.numerator = steps
                train_steps = 0

                while not done:
                    train_steps += 1
                    if config.stats.generate_states:
                        states.append(state0.numpy())
                    action0 = agent.get_action(state0)
                    next_state, reward, done, _ = self._env.step(action0.numpy())
                    train_ext_reward += reward

                    train_ext_rewards.append(train_ext_reward)

                    if self._preprocess is None:
                        state1 = torch.tensor(next_state, dtype=torch.float32)
                    else:
                        state1 = self._preprocess(next_state)

                    agent.train(state0, action0, state1, reward, done)
                    state0 = state1

                steps += train_steps

                print('Step {0:d} training [ext. reward {1:f} steps {2:d}]'.format(steps, train_ext_reward, train_steps))
                print(bar)

            agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))
            numpy.save('ddpg_{0}_{1}_{2:d}_re'.format(config.name, config.model, trial), numpy.array(train_ext_rewards[:step_limit]))

            if config.stats.generate_states:
                self.generate_states(states)

            if config.stats.collect_stats:
                action_list = torch.stack(action_list)
                value_list = torch.stack(value_list)

                numpy.save('ddpg_{0}_{1}_{2:d}_actions'.format(config.name, config.model, trial), action_list)
                numpy.save('ddpg_{0}_{1}_{2:d}_values'.format(config.name, config.model, trial), value_list)

    def run_forward_model(self, agent, trial):
        config = self._config
        forward_model = agent.get_motivation_module()

        step_limit = config.steps * 1e6
        steps = 0

        states = None
        if config.stats.generate_states:
            states = []
        if config.stats.collect_stats:
            states = torch.tensor(numpy.load('./{0:s}_states.npy'.format(self._env_name)), dtype=torch.float32)

        if config.load:
            agent.load(config.load)

            for i in range(5):
                self.test(self._env, agent, render=False, video=False)
        else:
            action_list = []
            value_list = []
            fm_error_list = []
            reward_list = []

            train_fm_errors = []
            train_ext_rewards = []
            train_int_rewards = []

            bar = ProgressBar(config.steps * 1e6, max_width=40)

            while steps < step_limit:
                if config.stats.collect_stats:
                    actions, values, fm_errors, rewards = self.fm_activations(self._env, agent, forward_model, states)
                    action_list.append(actions)
                    value_list.append(values)
                    fm_error_list.append(fm_errors)
                    reward_list.append(rewards)

                state0 = torch.tensor(self._env.reset(), dtype=torch.float32)
                done = False
                train_ext_reward = 0
                train_int_reward = 0
                train_fm_error = 0
                bar.numerator = steps
                train_steps = 0

                while not done:
                    train_steps += 1
                    if config.stats.generate_states:
                        states.append(state0.numpy())
                    action0 = agent.get_action(state0)
                    next_state, reward, done, _ = self._env.step(action0.numpy())
                    train_ext_reward += reward

                    state1 = torch.tensor(next_state, dtype=torch.float32)
                    agent.train(state0, action0, state1, reward, done)
                    forward_model.train(state0, action0, state1)

                    train_ext_rewards.append(train_ext_reward)
                    train_int_rewards.append(forward_model.reward(state0, action0, state1).item())
                    train_fm_errors.append(forward_model.error(state0, action0, state1).item())

                    state0 = state1

                steps += train_steps

                print('Step {0:d} training [ext. reward {1:f} int. reward {2:f} steps {3:d}]'.format(steps, train_ext_reward, train_int_reward, train_steps))
                print(bar)

            agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

            numpy.save('ddpg_{0}_{1}_{2:d}_re'.format(config.name, config.model, trial), numpy.array(train_ext_rewards[:step_limit]))
            numpy.save('ddpg_{0}_{1}_{2:d}_ri'.format(config.name, config.model, trial), numpy.array(train_int_rewards[:step_limit]))
            numpy.save('ddpg_{0}_{1}_{2:d}_fme'.format(config.name, config.model, trial), numpy.array(train_fm_errors[:step_limit]))

            if config.stats.generate_states:
                self.generate_states(states)

            if config.stats.collect_stats:
                action_list = torch.stack(action_list)
                value_list = torch.stack(value_list)
                fm_error_list = torch.stack(fm_error_list)
                reward_list = torch.stack(reward_list)

                numpy.save('ddpg_{0}_{1}_{2:d}_actions'.format(config.name, config.model, trial), action_list)
                numpy.save('ddpg_{0}_{1}_{2:d}_values'.format(config.name, config.model, trial), value_list)
                numpy.save('ddpg_{0}_{1}_{2:d}_prediction_errors'.format(config.name, config.model, trial), fm_error_list)
                numpy.save('ddpg_{0}_{1}_{2:d}_rewards'.format(config.name, config.model, trial), reward_list)

    def run_metalearner_model(self, agent, trial):
        config = self._config
        metacritic = agent.get_motivation_module()
        forward_model = metacritic.get_forward_model()

        step_limit = config.steps * 1e6
        steps = 0

        states = None
        if config.stats.generate_states:
            states = []
        if config.stats.collect_stats:
            states = torch.tensor(numpy.load('./{0:s}_states.npy'.format(self._env_name)), dtype=torch.float32)

        if config.load:
            agent.load(config.load)

            for i in range(5):
                self.test(self._env, agent, render=False, video=False)
        else:
            action_list = []
            value_list = []
            fm_error_list = []
            mc_error_list = []
            reward_list = []

            train_fm_errors = []
            train_mc_errors = []
            train_ext_rewards = []
            train_int_rewards = []

            bar = ProgressBar(config.steps * 1e6, max_width=40)

            while steps < step_limit:
                if config.stats.collect_stats:
                    actions, values, fm_errors, mc_errors, rewards = self.su_activations(self._env, agent, forward_model, metacritic, states)
                    action_list.append(actions)
                    value_list.append(values)
                    fm_error_list.append(fm_errors)
                    mc_error_list.append(mc_errors)
                    reward_list.append(rewards)

                state0 = torch.tensor(self._env.reset(), dtype=torch.float32)
                done = False
                train_fm_error = 0
                train_mc_error = 0
                train_ext_reward = 0
                train_int_reward = 0
                bar.numerator = steps
                train_steps = 0

                while not done:
                    train_steps += 1
                    if config.stats.generate_states:
                        states.append(state0.numpy())
                    action0 = agent.get_action(state0)
                    next_state, reward, done, _ = self._env.step(action0.numpy())
                    train_ext_reward += reward
                    state1 = torch.tensor(next_state, dtype=torch.float32)
                    agent.train(state0, action0, state1, reward, done)
                    metacritic.train(state0, action0, state1)

                    train_ext_rewards.append(train_ext_reward)
                    train_int_rewards.append(metacritic.reward(state0, action0, state1).item())
                    train_fm_errors.append(forward_model.error(state0, action0, state1).item())
                    train_mc_errors.append(metacritic.error(state0, action0).item())

                    state0 = state1

                steps += train_steps

                print('Episode {0:d} training [ext. reward {1:f} int. reward {2:f} steps {3:d}]'.format(steps, train_ext_reward, train_int_reward, train_steps))
                print(bar)

            agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

            numpy.save('ddpg_{0}_{1}_{2:d}_re'.format(config.name, config.model, trial), numpy.array(train_ext_rewards[:step_limit]))
            numpy.save('ddpg_{0}_{1}_{2:d}_ri'.format(config.name, config.model, trial), numpy.array(train_int_rewards[:step_limit]))
            numpy.save('ddpg_{0}_{1}_{2:d}_fme'.format(config.name, config.model, trial), numpy.array(train_fm_errors[:step_limit]))
            numpy.save('ddpg_{0}_{1}_{2:d}_mce'.format(config.name, config.model, trial), numpy.array(train_mc_errors[:step_limit]))

            if config.stats.generate_states:
                self.generate_states(states)

            if config.stats.collect_stats:
                action_list = torch.stack(action_list)
                value_list = torch.stack(value_list)
                fm_error_list = torch.stack(fm_error_list)
                mc_error_list = torch.stack(mc_error_list)
                reward_list = torch.stack(reward_list)

                numpy.save('ddpg_{0}_{1}_{2:d}_actions'.format(config.name, config.model, trial), action_list)
                numpy.save('ddpg_{0}_{1}_{2:d}_values'.format(config.name, config.model, trial), value_list)
                numpy.save('ddpg_{0}_{1}_{2:d}_prediction_errors'.format(config.name, config.model, trial), fm_error_list)
                numpy.save('ddpg_{0}_{1}_{2:d}_error_estimations'.format(config.name, config.model, trial), mc_error_list)
                numpy.save('ddpg_{0}_{1}_{2:d}_rewards'.format(config.name, config.model, trial), reward_list)

    @staticmethod
    def baseline_activations(agent, states):
        actions = agent.get_action(states)
        values = agent.get_value(states, actions)
        return actions, values

    @staticmethod
    def fm_activations(env, agent, forward_model, states):
        actions, values = ExperimentNoisyDDPG.baseline_activations(agent, states)
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

    @staticmethod
    def su_activations(env, agent, forward_model, metacritic, states):
        actions, values = ExperimentNoisyDDPG.baseline_activations(agent, states)
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

    def generate_states(self, states):
        kmeans = KMeans(n_clusters=2000, random_state=0).fit(states)
        states = numpy.stack(kmeans.cluster_centers_)
        states[:, 6] = numpy.round(states[:, 6])
        states[:, 7] = numpy.round(states[:, 7])
        numpy.save('{0:s}_states'.format(self._env_name), states)
