import time

import gym
import psutil
import pybulletgym
import numpy
import torch
# import umap
import umap
from etaprogress.progress import ProgressBar
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from scipy.spatial.distance import cdist
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer

from analytic.DOPAnalytic import DOPAnalytic
from analytic.RNDAnalytic import RNDAnalytic
from exploration.ContinuousExploration import GaussianExploration
from utils import stratify_sampling
from utils.RunningAverage import RunningAverageWindow


class ExperimentDDPG:
    def __init__(self, env_name, env, config):
        self._env_name = env_name
        self._env = env
        self._config = config
        self._preprocess = None
        self._reward_transform = None

    def add_preprocess(self, preprocess):
        self._preprocess = preprocess

    def add_reward_transform(self, reward_transform):
        self._reward_transform = reward_transform

    def transform_reward(self, reward):
        r = reward
        if self._reward_transform is not None:
            r = self._reward_transform(reward)

        return r

    def test(self, env, agent, render=False, video=False):
        config = self._config
        video_recorder = None

        for i in range(5):
            state0 = torch.tensor(env.reset(), dtype=torch.float32)
            done = False
            ext_rewards = 0
            steps = 0

            print("Test no.{0}".format(i))
            bar = ProgressBar(env.spec.max_episode_steps, max_width=40)
            if video:
                video_path = './videos/{0}_{1}_{2:d}.mp4'.format(config.name, config.model, i)
                video_recorder = VideoRecorder(env, video_path, enabled=video_path is not None)

            while not done:
                steps += 1
                if render:
                    env.render()
                if video:
                    env.render()
                    video_recorder.capture_frame()

                action = agent.get_action(state0)
                next_state, reward, done, _ = env.step(action.detach().numpy())
                next_state = torch.tensor(next_state, dtype=torch.float32)

                ext_rewards += reward
                state0 = next_state

                bar.numerator = steps
                print(bar)

            if video:
                video_recorder.close()

    def run_baseline(self, agent, trial):
        config = self._config
        trial = trial + config.shift
        step_limit = int(config.steps * 1e6)
        steps = 0

        steps_per_episode = []
        train_ext_rewards = []
        reward_avg = RunningAverageWindow(100)

        bar = ProgressBar(config.steps * 1e6, max_width=40)
        exploration = GaussianExploration(config.sigma, 0.01, config.steps * config.exploration_time * 1e6)

        while steps < step_limit:
            if self._preprocess is None:
                state0 = torch.tensor(self._env.reset(), dtype=torch.float32).unsqueeze(0)
            else:
                state0 = self._preprocess(self._env.reset())

            done = False
            train_ext_reward = 0
            train_steps = 0

            while not done:
                train_steps += 1
                action0 = exploration.explore(agent.get_action(state0))
                next_state, reward, done, _ = self._env.step(agent.convert_action(action0))
                reward = self.transform_reward(reward)
                train_ext_reward += reward

                if self._preprocess is None:
                    state1 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                else:
                    state1 = self._preprocess(next_state)

                reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
                mask = torch.tensor([done], dtype=torch.float32).unsqueeze(0)

                agent.train(state0, action0, state1, reward, mask)
                state0 = state1

            steps += train_steps
            if steps > step_limit:
                train_steps -= steps - step_limit
            bar.numerator = steps
            exploration.update(steps)

            reward_avg.update(train_ext_reward)
            steps_per_episode.append(train_steps)
            train_ext_rewards.append(train_ext_reward)

            print('Run {0:d} step {1:d} sigma {2:f} training [ext. reward {3:f} steps {4:d}] avg. ext. reward {5:f}'.format(
                trial, steps, exploration.sigma, train_ext_reward, train_steps, reward_avg.value().item()))
            print(bar)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Saving data...')
        save_data = {
            'steps': numpy.array(steps_per_episode),
            're': numpy.array(train_ext_rewards)
        }
        numpy.save('ddpg_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)

    def run_forward_model(self, agent, trial):
        config = self._config
        trial = trial + config.shift

        step_limit = int(config.steps * 1e6)
        steps = 0

        steps_per_episode = []
        train_fm_errors = []
        train_ext_rewards = []
        train_int_rewards = []
        reward_avg = RunningAverageWindow(100)
        step_avg = RunningAverageWindow(100)

        bar = ProgressBar(config.steps * 1e6, max_width=40)
        exploration = GaussianExploration(config.sigma, 0.01, config.steps * config.exploration_time * 1e6)

        while steps < step_limit:
            state0 = torch.tensor(self._env.reset(), dtype=torch.float32).unsqueeze(0)
            done = False
            train_ext_reward = 0
            train_int_reward = 0
            train_steps = 0

            while not done:
                action0 = exploration.explore(agent.get_action(state0))
                next_state, reward, done, _ = self._env.step(action0.squeeze(0).numpy())
                reward = self.transform_reward(reward)
                state1 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
                mask = torch.tensor([done], dtype=torch.float32).unsqueeze(0)

                agent.train(state0, action0, state1, reward, mask)
                train_steps += 1

                train_ext_reward += reward
                train_int_reward += agent.motivation.reward(state0, action0, state1).item()
                train_fm_error = agent.motivation.error(state0, action0, state1).item()
                train_fm_errors.append(train_fm_error)

                state0 = state1

            steps += train_steps
            if steps > step_limit:
                train_steps -= steps - step_limit
            bar.numerator = steps
            exploration.update(steps)

            reward_avg.update(train_ext_reward)
            step_avg.update(train_steps)
            steps_per_episode.append(train_steps)
            train_ext_rewards.append(train_ext_reward)
            train_int_rewards.append(train_int_reward)

            print('Run {0:d} step {1:d} sigma {2:f} training [ext. reward {3:f} int. reward {4:f} steps {5:d}] avg. ext. reward {6:f} avg. steps {7:f}'.format(
                trial, steps, exploration.sigma, train_ext_reward, train_int_reward, train_steps, reward_avg.value(), step_avg.value()))
            print(bar)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Saving data...')
        save_data = {
            'steps': numpy.array(steps_per_episode),
            're': numpy.array(train_ext_rewards),
            'ri': numpy.array(train_int_rewards),
            'fme': numpy.array(train_fm_errors[:step_limit])
        }
        numpy.save('ddpg_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)

    def run_forward_model_encoder(self, agent, trial):
        config = self._config
        trial = trial + config.shift

        step_limit = int(config.steps * 1e6)
        steps = 0

        states = []

        steps_per_episode = []
        train_fm_errors = []
        train_ext_rewards = []
        train_int_rewards = []
        reward_avg = RunningAverageWindow(100)
        step_avg = RunningAverageWindow(100)

        bar = ProgressBar(config.steps * 1e6, max_width=40)
        exploration = GaussianExploration(config.sigma, 0.01, config.steps * config.exploration_time * 1e6)

        while steps < step_limit:
            state0 = torch.tensor(self._env.reset(), dtype=torch.float32).unsqueeze(0)
            done = False
            train_ext_reward = 0
            train_int_reward = 0
            train_steps = 0

            while not done:
                train_steps += 1
                states.append(state0.squeeze(0))
                action0 = exploration.explore(agent.get_action(state0))
                next_state, reward, done, _ = self._env.step(action0.squeeze(0).numpy())
                reward = self.transform_reward(reward)
                state1 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

                agent.train(state0, action0, state1, reward, done)

                train_ext_reward += reward
                train_int_reward += agent.motivation.reward(state0, action0, state1).item()
                train_fm_error = agent.motivation.error(state0, action0, state1).item()
                train_fm_errors.append(train_fm_error)

                state0 = state1

            steps += train_steps
            if steps > step_limit:
                train_steps -= steps - step_limit
            bar.numerator = steps
            exploration.update(steps)

            reward_avg.update(train_ext_reward)
            step_avg.update(train_steps)
            steps_per_episode.append(train_steps)
            train_ext_rewards.append(train_ext_reward)
            train_int_rewards.append(train_int_reward)

            print('Run {0:d} step {1:d} sigma {2:f} training [ext. reward {3:f} int. reward {4:f} steps {5:d}] avg. ext. reward {6:f} avg. steps {7:f}'.format(
                trial, steps, exploration.sigma, train_ext_reward, train_int_reward, train_steps, reward_avg.value(), step_avg.value()))
            print(bar)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Calculating distance matrices')
        states = self.generate_states(torch.stack(states[:step_limit]), 500)
        state_dist = cdist(states.flatten(1), states.flatten(1), 'euclidean')
        index_list = numpy.argsort(numpy.linalg.norm(state_dist, axis=1))
        states = states[index_list]
        state_dist = cdist(states.flatten(1), states.flatten(1), 'euclidean')
        latent_states = agent.network.encoder(states).detach()
        latent_dist = torch.cdist(latent_states, latent_states)

        print('Saving data...')
        save_data = {
            'steps': numpy.array(steps_per_episode),
            're': numpy.array(train_ext_rewards),
            'ri': numpy.array(train_int_rewards),
            'fme': numpy.array(train_fm_errors[:step_limit]),
            'sdm': state_dist,
            'ldm': latent_dist.numpy()
        }
        numpy.save('ddpg_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)

    def run_inverse_model(self, agent, trial):
        config = self._config
        trial = trial + config.shift

        step_limit = int(config.steps * 1e6)
        steps = 0

        states = []

        steps_per_episode = []
        train_fm_errors = []
        train_ext_rewards = []
        train_int_rewards = []
        reward_avg = RunningAverageWindow(100)
        step_avg = RunningAverageWindow(100)

        bar = ProgressBar(config.steps * 1e6, max_width=40)
        exploration = GaussianExploration(config.sigma, 0.01, config.steps * config.exploration_time * 1e6)

        while steps < step_limit:
            state0 = torch.tensor(self._env.reset(), dtype=torch.float32).unsqueeze(0)
            done = False
            train_ext_reward = 0
            train_int_reward = 0
            train_steps = 0

            while not done:
                train_steps += 1
                states.append(state0.squeeze(0))
                action0 = exploration.explore(agent.get_action(state0))
                next_state, reward, done, _ = self._env.step(action0.squeeze(0).numpy())
                reward = self.transform_reward(reward)
                state1 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

                agent.train(state0, action0, state1, reward, done)

                train_ext_reward += reward
                train_int_reward += agent.motivation.reward(state0, action0, state1).item()
                train_fm_error = agent.motivation.error(state0, action0, state1).item()
                train_fm_errors.append(train_fm_error)

                state0 = state1

            steps += train_steps
            if steps > step_limit:
                train_steps -= steps - step_limit
            bar.numerator = steps
            exploration.update(steps)

            reward_avg.update(train_ext_reward)
            step_avg.update(train_steps)
            steps_per_episode.append(train_steps)
            train_ext_rewards.append(train_ext_reward)
            train_int_rewards.append(train_int_reward)

            print('Run {0:d} step {1:d} sigma {2:f} training [ext. reward {3:f} int. reward {4:f} steps {5:d}] avg. ext. reward {6:f} avg. steps {7:f}'.format(
                trial, steps, exploration.sigma, train_ext_reward, train_int_reward, train_steps, reward_avg.value(), step_avg.value()))
            print(bar)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Calculating distance matrices')
        states = self.generate_states(torch.stack(states[:step_limit]), 500)
        state_dist = cdist(states.flatten(1), states.flatten(1), 'euclidean')
        index_list = numpy.argsort(numpy.linalg.norm(state_dist, axis=1))
        states = states[index_list]
        state_dist = cdist(states.flatten(1), states.flatten(1), 'euclidean')
        latent_states = agent.network.encoder(states).detach()
        latent_dist = torch.cdist(latent_states, latent_states)

        print('Saving data...')
        save_data = {
            'steps': numpy.array(steps_per_episode),
            're': numpy.array(train_ext_rewards),
            'ri': numpy.array(train_int_rewards),
            'fme': numpy.array(train_fm_errors[:step_limit]),
            'sdm': state_dist,
            'ldm': latent_dist.numpy()
        }
        numpy.save('ddpg_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)

    def run_forward_inverse_model(self, agent, trial):
        config = self._config
        trial = trial + config.shift

        step_limit = int(config.steps * 1e6)
        steps = 0

        states = []

        steps_per_episode = []
        train_fm_errors = []
        train_im_errors = []
        train_ext_rewards = []
        train_int_rewards = []
        reward_avg = RunningAverageWindow(100)
        step_avg = RunningAverageWindow(100)

        bar = ProgressBar(config.steps * 1e6, max_width=40)
        exploration = GaussianExploration(config.sigma, 0.01, config.steps * config.exploration_time * 1e6)

        while steps < step_limit:
            state0 = torch.tensor(self._env.reset(), dtype=torch.float32).unsqueeze(0)
            done = False
            train_ext_reward = 0
            train_int_reward = 0
            train_steps = 0

            while not done:
                train_steps += 1
                states.append(state0.squeeze(0))
                action0 = exploration.explore(agent.get_action(state0))
                next_state, reward, done, _ = self._env.step(action0.squeeze(0).numpy())
                reward = self.transform_reward(reward)
                state1 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

                agent.train(state0, action0, state1, reward, done)

                train_ext_reward += reward
                train_int_reward += agent.motivation.reward(state0, action0, state1).item()
                train_fm_error, train_im_error = agent.motivation.error(state0, action0, state1)
                train_fm_errors.append(train_fm_error.item())
                train_im_errors.append(train_im_error.item())

                state0 = state1

            steps += train_steps
            if steps > step_limit:
                train_steps -= steps - step_limit
            bar.numerator = steps
            exploration.update(steps)

            reward_avg.update(train_ext_reward)
            step_avg.update(train_steps)
            steps_per_episode.append(train_steps)
            train_ext_rewards.append(train_ext_reward)
            train_int_rewards.append(train_int_reward)

            print('Run {0:d} step {1:d} sigma {2:f} training [ext. reward {3:f} int. reward {4:f} steps {5:d}] avg. ext. reward {6:f} avg. steps {7:f}'.format(
                trial, steps, exploration.sigma, train_ext_reward, train_int_reward, train_steps, reward_avg.value(), step_avg.value()))
            print(bar)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Calculating distance matrices')
        states = self.generate_states(torch.stack(states[:step_limit]), 500)
        state_dist = cdist(states.flatten(1), states.flatten(1), 'euclidean')
        index_list = numpy.argsort(numpy.linalg.norm(state_dist, axis=1))
        states = states[index_list]
        state_dist = cdist(states.flatten(1), states.flatten(1), 'euclidean')
        latent_states = agent.network.encoder(states).detach()
        latent_dist = torch.cdist(latent_states, latent_states)

        print('Saving data...')
        save_data = {
            'steps': numpy.array(steps_per_episode),
            're': numpy.array(train_ext_rewards),
            'ri': numpy.array(train_int_rewards),
            'fme': numpy.array(train_fm_errors[:step_limit]),
            'ime': numpy.array(train_im_errors[:step_limit]),
            'sdm': state_dist,
            'ldm': latent_dist.numpy()
        }
        numpy.save('ddpg_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)

    def run_vae_forward_model(self, agent, trial):
        config = self._config
        trial = trial + config.shift
        forward_model = agent.get_motivation_module()
        vae = forward_model.get_fm_network()

        step_limit = int(config.steps * 1e6)
        steps = 0

        states = None
        if config.check('generate_states'):
            states = []
        if config.check('collect_stats'):
            states = torch.tensor(numpy.load('./{0:s}_states.npy'.format(self._env_name)), dtype=torch.float32)

        action_list = []
        value_list = []
        fm_error_list = []
        reward_list = []

        steps_per_episode = []
        train_fm_errors = []
        train_ext_rewards = []
        train_int_rewards = []
        train_vae_losses = []
        reward_avg = RunningAverageWindow(100)
        step_avg = RunningAverageWindow(100)

        bar = ProgressBar(config.steps * 1e6, max_width=40)
        exploration = GaussianExploration(config.sigma, 0.01, config.steps * config.exploration_time * 1e6)

        while steps < step_limit:
            if config.check('collect_stats'):
                actions, values, fm_errors, rewards = self.fm_activations(self._env, agent, forward_model, states)
                action_list.append(actions)
                value_list.append(values)
                fm_error_list.append(fm_errors)
                reward_list.append(rewards)

            state0 = torch.tensor(self._env.reset(), dtype=torch.float32).unsqueeze(0)
            done = False
            train_ext_reward = 0
            train_int_reward = 0
            train_vae_loss = 0
            train_steps = 0

            while not done:
                train_steps += 1
                if config.check('generate_states'):
                    states.append(state0.numpy())
                action0 = exploration.explore(agent.get_action(state0))
                next_state, reward, done, _ = self._env.step(action0.squeeze(0).numpy())
                reward = self.transform_reward(reward)
                state1 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

                agent.train(state0, action0, state1, reward, done)
                forward_model.train(state0, action0, state1)

                train_ext_reward += reward
                train_int_reward += forward_model.reward(state0, action0, state1).item()
                train_vae_loss += vae.loss_function(state0, action0, state1).item()
                train_fm_error = forward_model.error(state0, action0, state1).item()
                train_fm_errors.append(train_fm_error)

                state0 = state1

            steps += train_steps
            if steps > step_limit:
                train_steps -= steps - step_limit
            bar.numerator = steps
            exploration.update(steps)

            reward_avg.update(train_ext_reward)
            step_avg.update(train_steps)
            steps_per_episode.append(train_steps)
            train_ext_rewards.append(train_ext_reward)
            train_int_rewards.append(train_int_reward)
            train_vae_losses.append(train_vae_loss)

            print('Run {0} step {1:d} sigma {2:f} training [ext. reward {3:f} int. reward {4:f} VAE loss {5:f} steps {6:d}] avg. ext. reward {7:f} avg. steps {8:f}'.format(
                trial, steps, exploration.sigma, train_ext_reward, train_int_reward, train_vae_loss, train_steps, reward_avg.value(), step_avg.value()))
            print(bar)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Saving data...')
        save_data = {
            'steps': numpy.array(steps_per_episode),
            're': numpy.array(train_ext_rewards),
            'ri': numpy.array(train_int_rewards),
            'fme': numpy.array(train_fm_errors[:step_limit]),
            'vl': numpy.array(train_vae_losses),
        }
        numpy.save('ddpg_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)

        if config.check('generate_states'):
            self.generate_states(states)

        if config.check('collect_stats'):
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
        trial = trial + config.shift

        step_limit = int(config.steps * 1e6)
        steps = 0

        steps_per_episode = []
        train_fm_errors = []
        train_mc_errors = []
        train_fm_rewards = []
        train_mc_rewards = []
        train_ext_rewards = []
        train_int_rewards = []
        reward_avg = RunningAverageWindow(100)
        step_avg = RunningAverageWindow(100)

        bar = ProgressBar(config.steps * 1e6, max_width=40)
        exploration = GaussianExploration(config.sigma, 0.01, config.steps * config.exploration_time * 1e6)

        while steps < step_limit:
            state0 = torch.tensor(self._env.reset(), dtype=torch.float32).unsqueeze(0)
            done = False
            train_ext_reward = 0
            train_int_reward = 0
            train_steps = 0

            while not done:
                train_steps += 1
                action0 = exploration.explore(agent.get_action(state0))
                next_state, reward, done, _ = self._env.step(action0.squeeze(0).numpy())
                reward = self.transform_reward(reward)
                state1 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

                pe_error, ps_error, pe_reward, ps_reward, int_reward = agent.motivation.raw_data(state0, action0, state1)
                train_ext_reward += reward
                train_int_reward += int_reward.item()
                train_fm_rewards.append(pe_reward.item())
                train_fm_errors.append(pe_error.item())
                train_mc_rewards.append(ps_reward.item())
                train_mc_errors.append(ps_error.item())

                agent.train(state0, action0, state1, reward, done)
                state0 = state1

            steps += train_steps
            if steps > step_limit:
                train_steps -= steps - step_limit
            bar.numerator = steps
            exploration.update(steps)

            reward_avg.update(train_ext_reward)
            step_avg.update(train_steps)
            steps_per_episode.append(train_steps)
            train_ext_rewards.append(train_ext_reward)
            train_int_rewards.append(train_int_reward)

            print('Run {0:d} step {1:d} sigma {2:f} training [ext. reward {3:f} int. reward {4:f} steps {5:d}] avg. ext. reward {6:f} avg. steps {7:f}'.format(
                trial, steps, exploration.sigma, train_ext_reward, train_int_reward, train_steps, reward_avg.value(), step_avg.value()))
            print(bar)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Saving data...')
        save_data = {
            'steps': numpy.array(steps_per_episode),
            're': numpy.array(train_ext_rewards),
            'ri': numpy.array(train_int_rewards),
            'fme': numpy.array(train_fm_errors[:step_limit]),
            'fmr': numpy.array(train_fm_rewards[:step_limit]),
            'mce': numpy.array(train_mc_errors[:step_limit]),
            'mcr': numpy.array(train_mc_rewards[:step_limit])
        }
        numpy.save('ddpg_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)

    def run_metalearner_rnd_model(self, agent, trial):
        config = self._config
        trial = trial + config.shift

        step_limit = int(config.steps * 1e6)
        steps = 0

        steps_per_episode = []
        train_fm_errors = []
        train_mc_errors = []
        train_fm_rewards = []
        train_mc_rewards = []
        train_ext_rewards = []
        train_int_rewards = []
        reward_avg = RunningAverageWindow(100)
        step_avg = RunningAverageWindow(100)

        bar = ProgressBar(config.steps * 1e6, max_width=40)
        exploration = GaussianExploration(config.sigma, 0.01, config.steps * config.exploration_time * 1e6)

        while steps < step_limit:
            state0 = torch.tensor(self._env.reset(), dtype=torch.float32).unsqueeze(0)
            done = False
            train_ext_reward = 0
            train_int_reward = 0
            train_steps = 0

            while not done:
                train_steps += 1
                action0 = exploration.explore(agent.get_action(state0))
                next_state, reward, done, _ = self._env.step(action0.squeeze(0).numpy())
                reward = self.transform_reward(reward)
                state1 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

                pe_error, ps_error, pe_reward, ps_reward, int_reward = agent.motivation.raw_data(state0)
                train_ext_reward += reward
                train_int_reward += int_reward.item()
                train_fm_rewards.append(pe_reward.item())
                train_fm_errors.append(pe_error.item())
                train_mc_rewards.append(ps_reward.item())
                train_mc_errors.append(ps_error.item())

                agent.train(state0, action0, state1, reward, done)
                state0 = state1

            steps += train_steps
            if steps > step_limit:
                train_steps -= steps - step_limit
            bar.numerator = steps
            exploration.update(steps)

            reward_avg.update(train_ext_reward)
            step_avg.update(train_steps)
            steps_per_episode.append(train_steps)
            train_ext_rewards.append(train_ext_reward)
            train_int_rewards.append(train_int_reward)

            print('Run {0:d} step {1:d} sigma {2:f} training [ext. reward {3:f} int. reward {4:f} steps {5:d}] avg. ext. reward {6:f} avg. steps {7:f}'.format(
                trial, steps, exploration.sigma, train_ext_reward, train_int_reward, train_steps, reward_avg.value(), step_avg.value()))
            print(bar)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Saving data...')
        save_data = {
            'steps': numpy.array(steps_per_episode),
            're': numpy.array(train_ext_rewards),
            'ri': numpy.array(train_int_rewards),
            'fme': numpy.array(train_fm_errors[:step_limit]),
            'fmr': numpy.array(train_fm_rewards[:step_limit]),
            'mce': numpy.array(train_mc_errors[:step_limit]),
            'mcr': numpy.array(train_mc_rewards[:step_limit])
        }
        numpy.save('ddpg_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)

    def run_m2_model(self, agent, trial):
        config = self._config
        trial = trial + config.shift

        step_limit = int(config.steps * 1e6)
        steps = 0

        states = []

        steps_per_episode = []
        train_fm_errors = []
        train_ext_rewards = []
        train_int_rewards = []
        train_m2_weight = []
        reward_avg = RunningAverageWindow(100)
        step_avg = RunningAverageWindow(100)

        bar = ProgressBar(config.steps * 1e6, max_width=40)
        exploration = GaussianExploration(config.sigma, 0.01, config.steps * config.exploration_time * 1e6)

        while steps < step_limit:
            state0 = torch.tensor(self._env.reset(), dtype=torch.float32).unsqueeze(0)
            im0 = torch.zeros((1, 1), dtype=torch.float32)
            error0 = torch.zeros((1, 1), dtype=torch.float32)
            done = False
            train_ext_reward = 0
            train_int_reward = 0
            train_steps = 0

            while not done:
                train_steps += 1
                states.append(state0.squeeze(0))
                action0 = exploration.explore(agent.get_action(state0))
                next_state, reward, done, _ = self._env.step(action0.squeeze(0).numpy())
                reward = self.transform_reward(reward)
                state1 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

                weight = agent.motivation.weight(agent.compose_gate_state(im0, error0))
                im1 = agent.motivation.reward(state0, action0, weight, state1)

                train_ext_reward += reward
                train_int_reward += im1.item()
                error1 = agent.network.forward_model.error(state0, action0, state1)
                train_fm_errors.append(error1.item())
                train_m2_weight.append(weight.squeeze(0).numpy())

                agent.train(state0, action0, state1, im0, error0, weight, im1, error1, reward, done)

                state0 = state1
                im0 = im1
                error0 = error1

            steps += train_steps
            if steps > step_limit:
                train_steps -= steps - step_limit
            bar.numerator = steps
            exploration.update(steps)

            reward_avg.update(train_ext_reward)
            step_avg.update(train_steps)
            steps_per_episode.append(train_steps)
            train_ext_rewards.append(train_ext_reward)
            train_int_rewards.append(train_int_reward)

            print('Run {0:d} step {1:d} sigma {2:f} training [ext. reward {3:f} int. reward {4:f} steps {5:d}] avg. ext. reward {6:f} avg. steps {7:f}'.format(trial, steps, exploration.sigma,
                                                                                                                                                               train_ext_reward, train_int_reward,
                                                                                                                                                               train_steps, reward_avg.value(),
                                                                                                                                                               step_avg.value()))
            print(bar)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Saving data...')
        save_data = {
            'steps': numpy.array(steps_per_episode),
            're': numpy.array(train_ext_rewards),
            'ri': numpy.array(train_int_rewards),
            'fme': numpy.array(train_fm_errors[:step_limit]),
            'm2w': numpy.array(train_m2_weight[:step_limit])
        }
        numpy.save('ddpg_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)

    def run_rnd_model(self, agent, trial):
        config = self._config
        trial = trial + config.shift

        step_limit = int(config.steps * 1e6)
        steps = 0

        steps_per_episode = []
        train_fm_errors = []
        train_ext_rewards = []
        train_int_rewards = []
        reward_avg = RunningAverageWindow(100)

        bar = ProgressBar(config.steps * 1e6, max_width=40)
        exploration = GaussianExploration(config.sigma, 0.01, config.steps * config.exploration_time * 1e6)

        while steps < step_limit:
            state0 = torch.tensor(self._env.reset(), dtype=torch.float32).unsqueeze(0)
            done = False
            train_ext_reward = 0
            train_int_reward = 0
            train_steps = 0

            while not done:
                agent.motivation.update_state_average(state0)
                action0 = exploration.explore(agent.get_action(state0))
                next_state, reward, done, _ = self._env.step(action0.squeeze(0).numpy())
                reward = self.transform_reward(reward)
                state1 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
                mask = torch.tensor([done], dtype=torch.float32).unsqueeze(0)

                agent.train(state0, action0, state1, reward, mask)
                train_steps += 1

                train_ext_reward += reward.item()
                train_int_reward += agent.motivation.reward(state0).item()
                train_fm_error = agent.motivation.error(state0).item()
                train_fm_errors.append(train_fm_error)

                state0 = state1

            steps += train_steps
            if steps > step_limit:
                train_steps -= steps - step_limit
            bar.numerator = steps
            exploration.update(steps)

            reward_avg.update(train_ext_reward)
            steps_per_episode.append(train_steps)
            train_ext_rewards.append(train_ext_reward)
            train_int_rewards.append(train_int_reward)

            print('Run {0:d} step {1:d} sigma {2:f} training [ext. reward {3:f} int. reward {4:f} steps {5:d} ({6:f})] avg. ext. reward {7:f}'.format(
                trial, steps, exploration.sigma, train_ext_reward, train_int_reward, train_steps, train_int_reward / train_steps, reward_avg.value().item()))
            print(bar)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Saving data...')
        save_data = {
            'steps': numpy.array(steps_per_episode),
            're': numpy.array(train_ext_rewards),
            'ri': numpy.array(train_int_rewards),
            'fme': numpy.array(train_fm_errors[:step_limit])
        }
        numpy.save('ddpg_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)

    def run_qrnd_model(self, agent, trial):
        config = self._config
        trial = trial + config.shift

        step_limit = int(config.steps * 1e6)
        steps = 0

        steps_per_episode = []
        train_fm_errors = []
        train_ext_rewards = []
        train_int_rewards = []
        reward_avg = RunningAverageWindow(100)
        step_avg = RunningAverageWindow(100)

        bar = ProgressBar(config.steps * 1e6, max_width=40)
        exploration = GaussianExploration(config.sigma, 0.01, config.steps * config.exploration_time * 1e6)

        while steps < step_limit:
            state0 = torch.tensor(self._env.reset(), dtype=torch.float32).unsqueeze(0)
            done = False
            train_ext_reward = 0
            train_int_reward = 0
            train_steps = 0

            while not done:
                agent.motivation.update_state_average(state0)
                action0 = exploration.explore(agent.get_action(state0))
                next_state, reward, done, _ = self._env.step(action0.squeeze(0).numpy())
                reward = self.transform_reward(reward)
                state1 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
                mask = torch.tensor([done], dtype=torch.float32).unsqueeze(0)

                agent.train(state0, action0, state1, reward, mask)
                train_steps += 1

                train_ext_reward += reward.item()
                train_int_reward += agent.motivation.reward(state0, action0).item()
                train_fm_error = agent.motivation.error(state0, action0).item()
                train_fm_errors.append(train_fm_error)

                state0 = state1

            steps += train_steps
            if steps > step_limit:
                train_steps -= steps - step_limit
            bar.numerator = steps
            exploration.update(steps)

            reward_avg.update(train_ext_reward)
            step_avg.update(train_steps)
            steps_per_episode.append(train_steps)
            train_ext_rewards.append(train_ext_reward)
            train_int_rewards.append(train_int_reward)

            print('Run {0:d} step {1:d} sigma {2:f} training [ext. reward {3:f} int. reward {4:f} steps {5:d} ({6:f})] avg. ext. reward {7:f}'.format(
                trial, steps, exploration.sigma, train_ext_reward, train_int_reward, train_steps, train_int_reward / train_steps, reward_avg.value().item()))
            print(bar)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Saving data...')
        save_data = {
            'steps': numpy.array(steps_per_episode),
            're': numpy.array(train_ext_rewards),
            'ri': numpy.array(train_int_rewards),
            'fme': numpy.array(train_fm_errors[:step_limit])
        }
        numpy.save('ddpg_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)

    def run_dop_model(self, agent, trial):
        config = self._config
        trial = trial + config.shift

        step_limit = int(config.steps * 1e6)
        steps = 0

        steps_per_episode = []
        train_fm_errors = []
        train_ext_rewards = []
        train_int_rewards = []
        train_head_index = []
        reward_avg = RunningAverageWindow(100)
        step_avg = RunningAverageWindow(100)

        bar = ProgressBar(config.steps * 1e6, max_width=40)
        exploration = GaussianExploration(config.sigma, 0.01, config.steps * config.exploration_time * 1e6)

        while steps < step_limit:
            state0 = torch.tensor(self._env.reset(), dtype=torch.float32).unsqueeze(0)
            done = False
            train_ext_reward = 0
            train_int_reward = 0
            train_steps = 0
            head_index_density = numpy.zeros(4)

            while not done:
                agent.motivation.update_state_average(state0)
                action0, head_index = agent.get_action(state0)
                print(action0)
                next_state, reward, done, _ = self._env.step(action0.squeeze(0).numpy())
                reward = self.transform_reward(reward)
                state1 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
                mask = torch.tensor([done], dtype=torch.float32).unsqueeze(0)

                agent.train(state0, action0, state1, reward, mask)
                train_steps += 1

                train_ext_reward += reward.item()
                train_int_reward += agent.motivation.reward(state0, action0).item()
                train_fm_error = agent.motivation.error(state0, action0).item()
                train_fm_errors.append(train_fm_error)
                head_index_density[head_index.item()] += 1

                state0 = state1

            steps += train_steps
            if steps > step_limit:
                train_steps -= steps - step_limit
            bar.numerator = steps
            exploration.update(steps)

            reward_avg.update(train_ext_reward)
            step_avg.update(train_steps)
            steps_per_episode.append(train_steps)
            train_ext_rewards.append(train_ext_reward)
            train_int_rewards.append(train_int_reward)
            train_head_index.append(head_index_density)

            print('Run {0:d} step {1:d} sigma {2:f} training [ext. reward {3:f} int. reward {4:f} steps {5:d} ({6:f})] avg. ext. reward {7:f} density {8:s}'.format(
                trial, steps, exploration.sigma, train_ext_reward, train_int_reward, train_steps, train_int_reward / train_steps, reward_avg.value().item(), numpy.array2string(head_index_density)))
            print(bar)

        print('Saving agent...')
        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Running analysis...')
        states, actions, head_indices = DOPAnalytic.head_analyze(self._env, agent)

        print('Saving data...')
        save_data = {
            'steps': numpy.array(steps_per_episode),
            're': numpy.array(train_ext_rewards),
            'ri': numpy.array(train_int_rewards),
            'fme': numpy.array(train_fm_errors[:step_limit]),
            'hid': numpy.stack(train_head_index),
            'ts': states,
            'ta': actions,
            'th': head_indices,
        }
        numpy.save('ddpg_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)

    def run_dop2_model(self, agent, trial):
        config = self._config
        trial = trial + config.shift

        step_limit = int(config.steps * 1e6)
        steps = 0

        steps_per_episode = []
        train_fm_errors = []
        train_ext_rewards = []
        train_int_rewards = []
        train_head_index = []
        train_accuracy = []
        reward_avg = RunningAverageWindow(100)
        step_avg = RunningAverageWindow(100)

        bar = ProgressBar(config.steps * 1e6, max_width=80)
        exploration = GaussianExploration(config.sigma, 0.01, config.steps * config.exploration_time * 1e6)

        while steps < step_limit:
            state0 = torch.tensor(self._env.reset(), dtype=torch.float32).unsqueeze(0)
            done = False
            train_ext_reward = 0
            train_int_reward = 0
            train_steps = 0
            head_index_density = numpy.zeros(4)
            accuracy_per_episode = 0

            while not done:
                agent.motivation.update_state_average(state0)
                action0, head_index, arbiter_accuracy = agent.get_action(state0)
                next_state, reward, done, _ = self._env.step(action0.squeeze(0).numpy())
                reward = self.transform_reward(reward)
                state1 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
                mask = torch.tensor([done], dtype=torch.float32).unsqueeze(0)

                agent.train(state0, action0, state1, reward, mask)
                train_steps += 1

                train_ext_reward += reward.item()
                train_int_reward += agent.motivation.reward(state0, action0).item()
                train_fm_error = agent.motivation.error(state0, action0).item()
                train_fm_errors.append(train_fm_error)
                head_index_density[head_index.item()] += 1
                accuracy_per_episode += arbiter_accuracy
                state0 = state1

            steps += train_steps
            if steps > step_limit:
                train_steps -= steps - step_limit
            bar.numerator = steps
            exploration.update(steps)

            reward_avg.update(train_ext_reward)
            step_avg.update(train_steps)
            steps_per_episode.append(train_steps)
            train_ext_rewards.append(train_ext_reward)
            train_int_rewards.append(train_int_reward)
            train_head_index.append(head_index_density)
            train_accuracy.append(accuracy_per_episode / train_steps)

            print('Run {0:d} step {1:d} sigma {2:f} training [ext. reward {3:f} int. reward {4:f} steps {5:d} ({6:f})] avg. ext. reward {7:f} accuracy {8:.2f} density {9:s}'.format(
                trial, steps, exploration.sigma, train_ext_reward, train_int_reward, train_steps, train_int_reward / train_steps, reward_avg.value().item(), accuracy_per_episode / train_steps,
                numpy.array2string(head_index_density)))
            print(bar)

        print('Saving agent...')
        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Running analysis...')
        states, actions, head_indices = DOPAnalytic.head_analyze(self._env, agent)

        print('Saving data...')
        save_data = {
            'steps': numpy.array(steps_per_episode),
            're': numpy.array(train_ext_rewards),
            'ri': numpy.array(train_int_rewards),
            'fme': numpy.array(train_fm_errors[:step_limit]),
            'hid': numpy.stack(train_head_index),
            'aa': numpy.array(train_accuracy),
            'ts': states,
            'ta': actions,
            'th': head_indices,
        }
        numpy.save('ddpg_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)

    @staticmethod
    def baseline_activations(agent, states):
        actions = agent.get_action(states)
        values = agent.get_value(states, actions)
        return actions, values

    @staticmethod
    def fm_activations(env, agent, forward_model, states):
        actions, values = ExperimentDDPG.baseline_activations(agent, states)
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
    def su_activations(env, agent, metacritic, states):
        actions, values = ExperimentDDPG.baseline_activations(agent, states)
        next_states = []
        env.reset()
        for i in range(states.shape[0]):
            env.set_state(states[i].numpy())
            next_state, _, _, _ = env.step(actions[i].numpy())
            next_states.append(torch.tensor(next_state))
        next_states = torch.stack(next_states)
        fm_errors = metacritic.error(states, actions, next_states)
        mc_errors = metacritic.error_estimate(states, actions)
        rewards = metacritic.reward(states, actions, next_states)

        return actions, values, fm_errors, mc_errors, rewards

    @staticmethod
    def generate_states(states, n_clusters):
        # initial_centers = kmeans_plusplus_initializer(states, n_clusters).initialize()
        # kmeans_instance = kmeans(states, initial_centers)
        # kmeans_instance.process()
        # final_centers = kmeans_instance.get_centers()
        # states = numpy.stack(final_centers)

        stratify_size = states.shape[0] // n_clusters
        samples = stratify_sampling(states, n_clusters, [stratify_size] * n_clusters)

        return samples
