import time

import numpy
import torch
from etaprogress.progress import ProgressBar
from gym.wrappers.monitoring.video_recorder import VideoRecorder


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
            video_recorder = VideoRecorder(self._env, video_path, enabled=video_path is not None, fps=15)
            state0 = torch.tensor(self._env.reset(), dtype=torch.float32).unsqueeze(0).to(config.device)
            done = False

            while not done:
                self._env.render()
                video_recorder.capture_frame()
                action0 = agent.get_action(state0, True)
                next_state, reward, done, info = self._env.step(action0.item())
                state0 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(config.device)
            video_recorder.close()

    def run_baseline(self, agent, trial):
        config = self._config
        trial = trial + config.shift
        step_limit = int(config.steps * 1e6)
        steps = 0
        bar = ProgressBar(step_limit, max_width=40)

        train_ext_rewards = []
        reward_buffer = numpy.zeros(100)
        reward_buffer_index = 0

        while steps < step_limit:
            state0 = self.process_state(self._env.reset())
            done = False
            train_ext_reward = 0
            train_steps = 0

            while not done:
                a, action0, log_prob = agent.get_action(state0)
                next_state, reward, done, info = self._env.step(a)

                if isinstance(reward, numpy.ndarray):
                    reward = reward[0]

                state1 = self.process_state(next_state)
                mask = 1
                if done:
                    mask = 0
                agent.train(state0, action0.unsqueeze(0), log_prob.unsqueeze(0), state1, reward, mask)
                state0 = state1

                train_ext_reward += reward
                train_steps += 1

            if steps + train_steps > step_limit:
                train_steps = step_limit - steps
            steps += train_steps
            bar.numerator = steps

            train_ext_rewards.append([train_steps, train_ext_reward])

            reward_buffer[reward_buffer_index] = train_ext_reward
            reward_buffer_index += 1
            if reward_buffer_index == 100:
                reward_buffer_index = 0

            print('Run {0:d} step {1:d} training [ext. reward {2:f} steps {3:d} mean reward {4:f}]'.format(trial, steps, train_ext_reward, train_steps, reward_buffer.mean()))
            print(bar)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Saving data...')
        save_data = {
            're': numpy.array(train_ext_rewards)
        }
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)

    def run_forward_model(self, agent, trial):
        config = self._config
        trial = trial + config.shift
        forward_model = agent.get_motivation_module()

        step_limit = int(config.steps * 1e6)
        steps = 0

        train_fm_errors = []
        train_ext_rewards = []
        train_int_rewards = []

        bar = ProgressBar(config.steps * 1e6, max_width=40)

        while steps < step_limit:
            state0 = self.process_state(self._env.reset())
            done = False
            train_ext_reward = 0
            train_int_reward = 0
            train_steps = 0

            while not done:
                train_steps += 1
                action0 = agent.get_action(state0)
                next_state, reward, done, info = self._env.step(action0.item())
                state1 = self.process_state(next_state)

                agent.train(state0, action0, state1, reward, done)

                train_ext_reward += reward
                train_int_reward += forward_model.reward(state0, action0, state1).item()
                train_fm_error = forward_model.error(state0, action0, state1).item()
                train_fm_errors.append(train_fm_error)

                state0 = state1

            steps += train_steps
            if steps > step_limit:
                train_steps -= steps - step_limit
            bar.numerator = steps

            train_ext_rewards.append([train_steps, train_ext_reward])
            train_int_rewards.append([train_steps, train_int_reward])

            print('Run {0:d} step {1:d} training [ext. reward {2:f} int. reward {3:f} steps {4:d}]'.format(trial, steps, train_ext_reward, train_int_reward, train_steps))
            print(bar)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Saving data...')
        save_data = {
            're': numpy.array(train_ext_rewards),
            'ri': numpy.array(train_int_rewards),
            'fme': numpy.array(train_fm_errors[:step_limit])
        }
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)