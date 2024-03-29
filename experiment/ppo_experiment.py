import time

import numpy
import torch
from etaprogress.progress import ProgressBar
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from utils.RunningAverage import RunningAverageWindow, StepCounter


class ExperimentPPO:
    def __init__(self, env_name, env, config):
        self._env_name = env_name
        self._env = env
        self._config = config
        self._preprocess = None

    def add_preprocess(self, preprocess):
        self._preprocess = preprocess

    def process_state(self, state):
        if self._preprocess is None:
            processed_state = torch.tensor(state, dtype=torch.float32).to(self._config.device)
        else:
            processed_state = self._preprocess(state).to(self._config.device)

        return processed_state.unsqueeze(0)

    def test(self, agent):
        config = self._config

        for i in range(3):
            video_path = 'ppo_{0}_{1}_{2:d}.mp4'.format(config.name, config.model, i)
            video_recorder = VideoRecorder(self._env, video_path, enabled=video_path is not None)
            state0 = torch.tensor(self._env.reset(), dtype=torch.float32).unsqueeze(0).to(config.device)
            done = False

            while not done:
                self._env.render()
                video_recorder.capture_frame()
                _, action0, _ = agent.get_action(state0)
                next_state, reward, done, info = self._env.step(agent.convert_action(action0.cpu()))
                state0 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(config.device)
            video_recorder.close()

    def run_baseline(self, agent, trial):

        config = self._config
        trial = trial + config.shift
        step_limit = int(config.steps * 1e6)
        steps = 0
        bar = ProgressBar(step_limit, max_width=80)

        steps_per_episode = []
        train_ext_rewards = []
        reward_avg = RunningAverageWindow(100)
        # prob_avg = RunningAverageWindow(1000, 2)
        # value_avg = RunningAverageWindow(1000)

        while steps < step_limit:
            state0 = self.process_state(self._env.reset())
            done = False
            train_ext_reward = 0
            train_steps = 0

            while not done:
                value, action0, probs0 = agent.get_action(state0)
                # value_avg.update(value.numpy())
                # prob_avg.update(probs0.numpy())
                next_state, reward, done, info = self._env.step(agent.convert_action(action0.cpu()))
                state1 = self.process_state(next_state)

                if isinstance(reward, numpy.ndarray):
                    reward = reward[0]
                reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
                mask = torch.tensor([done], dtype=torch.float32).unsqueeze(0)

                agent.train(state0, value, action0, probs0, state1, reward, mask)
                state0 = state1

                if info is not None and 'raw_score' in info:
                    train_ext_reward += info['raw_score']
                else:
                    train_ext_reward += reward.item()
                # train_ext_reward += reward
                train_steps += 1

            if steps + train_steps > step_limit:
                train_steps = step_limit - steps
            steps += train_steps
            bar.numerator = steps

            steps_per_episode.append(train_steps)
            train_ext_rewards.append(train_ext_reward)
            reward_avg.update(train_ext_reward)

            # print('Run {0:d} step {1:d} training [ext. reward {2:f} steps {3:d} mean reward {4:f}] prob {5:s} value {6:f}'.format(trial, steps, train_ext_reward, train_steps, reward_avg.value().item(), numpy.array2string(prob_avg.value()), value_avg.value().item()))
            print('Run {0:d} step {1:d} training [ext. reward {2:f} steps {3:d} mean reward {4:f}]'.format(trial, steps, train_ext_reward, train_steps, reward_avg.value().item()))
            print(bar)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Saving data...')
        save_data = {
            'steps': numpy.array(steps_per_episode),
            're': numpy.array(train_ext_rewards)
        }
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)

    def run_rnd_model(self, agent, trial):
        config = self._config
        trial = trial + config.shift

        step_limit = int(config.steps * 1e6)
        steps = 0

        steps_per_episode = []
        train_fm_errors = []
        train_ext_rewards = []
        train_int_rewards = []

        bar = ProgressBar(step_limit, max_width=80)
        reward_avg = RunningAverageWindow(100)

        while steps < step_limit:
            state0 = self.process_state(self._env.reset())
            done = False
            train_ext_reward = 0
            train_int_reward = 0
            train_steps = 0

            while not done:
                train_steps += 1
                value, action0, probs0 = agent.get_action(state0)
                agent.motivation.update_state_average(state0)
                next_state, reward, done, info = self._env.step(agent.convert_action(action0.cpu()))
                state1 = self.process_state(next_state)
                ext_reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
                int_reward = agent.motivation.reward(state0).cpu()
                reward = torch.cat([ext_reward, int_reward], dim=1)
                mask = torch.tensor([done], dtype=torch.float32).unsqueeze(0)

                agent.train(state0, value, action0, probs0, state1, reward, mask)

                train_ext_reward += ext_reward.item()
                train_int_reward += int_reward.item()
                train_fm_error = agent.motivation.error(state0).item()
                train_fm_errors.append(train_fm_error)

                state0 = state1

            steps += train_steps
            if steps > step_limit:
                train_steps -= steps - step_limit
            bar.numerator = steps

            steps_per_episode.append(train_steps)
            train_ext_rewards.append(train_ext_reward)
            train_int_rewards.append(train_int_reward)
            reward_avg.update(train_ext_reward)

            print('Run {0:d} step {1:d} training [ext. reward {2:f} int. reward {3:f} steps {4:d} ({5:f})  mean reward {6:f}]'.format(
                trial, steps, train_ext_reward, train_int_reward, train_steps, train_int_reward / train_steps, reward_avg.value()))
            print(bar)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Saving data...')
        save_data = {
            'steps': numpy.array(steps_per_episode),
            're': numpy.array(train_ext_rewards),
            'ri': numpy.array(train_int_rewards),
            'fme': numpy.array(train_fm_errors[:step_limit])
        }
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)

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

        bar = ProgressBar(step_limit, max_width=80)
        reward_avg = RunningAverageWindow(100)

        while steps < step_limit:
            head_index_density = numpy.zeros(4)
            state0 = self.process_state(self._env.reset())
            done = False
            train_ext_reward = 0
            train_int_reward = 0
            train_error = 0
            train_steps = 0

            while not done:
                train_steps += 1
                value, action0, probs0 = agent.get_action(state0)
                agent.motivation.update_state_average(state0)
                action0, head_index = action0
                next_state, reward, done, info = self._env.step(agent.convert_action(action0))
                state1 = self.process_state(next_state)
                reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
                mask = torch.tensor([done], dtype=torch.float32).unsqueeze(0)

                agent.train(state0, value, action0, probs0, state1, reward, mask)

                train_ext_reward += reward.item()
                train_int_reward += agent.motivation.reward(state0, probs0).item()
                train_fm_error = agent.motivation.error(state0, probs0).item()
                train_error += train_fm_error
                train_fm_errors.append(train_fm_error)
                head_index_density[head_index.item()] += 1

                state0 = state1

            steps += train_steps
            if steps > step_limit:
                train_steps -= steps - step_limit
            bar.numerator = steps

            steps_per_episode.append(train_steps)
            train_ext_rewards.append(train_ext_reward)
            train_int_rewards.append(train_int_reward)
            train_head_index.append(head_index_density)
            reward_avg.update(train_ext_reward)

            print('Run {0:d} step {1:d} training [ext. reward {2:f} error {3:f} steps {4:d} ({5:f} err/step) mean reward {6:f} density {7:s}]'.format(
                trial, steps, train_ext_reward, train_error, train_steps, train_error / train_steps, reward_avg.value(), numpy.array2string(head_index_density)))
            print(bar)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Saving data...')
        save_data = {
            'steps': numpy.array(steps_per_episode),
            're': numpy.array(train_ext_rewards),
            'ri': numpy.array(train_int_rewards),
            'fme': numpy.array(train_fm_errors[:step_limit]),
            'hid': numpy.stack(train_head_index)
        }
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)
