import time

import numpy
import torch
from etaprogress.progress import ProgressBar
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from utils.RunningAverage import RunningAverageWindow


class ExperimentNEnvPPO:
    def __init__(self, env_name, env_list, config):
        self._env_name = env_name
        self._env = env_list[0]
        self._env_list = env_list
        self._config = config
        self._preprocess = None

        self._config.steps *= self._config.n_env
        self._config.batch_size *= self._config.n_env
        self._config.trajectory_size *= self._config.n_env

        print('Total steps: {0:.2f}M'.format(self._config.steps))
        print('Total batch size: {0:d}'.format(self._config.batch_size))
        print('Total trajectory size: {0:d}'.format(self._config.trajectory_size))

    def add_preprocess(self, preprocess):
        self._preprocess = preprocess

    def process_state(self, state):
        if self._preprocess is None:
            processed_state = torch.tensor(state, dtype=torch.float32).to(self._config.device)
        else:
            processed_state = self._preprocess(state).to(self._config.device)

        return processed_state

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
        n_env = config.n_env
        trial = trial + config.shift
        step_limit = int(config.steps * 1e6)
        steps = 0
        bar = ProgressBar(step_limit, max_width=40)

        train_ext_rewards = []
        train_ext_reward = [0] * n_env
        train_steps = [0] * n_env
        reward_avg = RunningAverageWindow(100)

        s = [None] * n_env
        v = [None] * n_env
        ns = [None] * n_env
        r = [None] * n_env
        d = [None] * n_env

        for i in range(n_env):
            s[i] = self._env_list[i].reset()

        state0 = self.process_state(numpy.stack(s))

        while steps < step_limit:
            value, action0, probs0 = agent.get_action(state0)

            for i in range(n_env):
                next_state, reward, done, info = self._env_list[i].step(agent.convert_action(action0[i]))
                mask = 1
                if done:
                    mask = 0

                # if 'raw_score' in info:
                #     train_ext_reward[i] += info['raw_score']
                # else:
                #     train_ext_reward[i] += reward
                train_ext_reward[i] += reward
                train_steps[i] += 1
                ns[i] = next_state
                r[i] = reward
                d[i] = mask

                if done:
                    if steps + train_steps[i] > step_limit:
                        train_steps[i] = step_limit - steps
                    steps += train_steps[i]

                    train_ext_rewards.append([train_steps[i], train_ext_reward[i]])
                    bar.numerator = steps

                    print('Run {0:d} step {1:d} training [ext. reward {2:f} steps {3:d}] avg. reward {4:f}'.format(trial, steps, train_ext_reward[i], train_steps[i], reward_avg.value()))
                    print(bar)

                    train_ext_reward[i] = 0
                    train_steps[i] = 0

                    s[i] = self._env_list[i].reset()
                else:
                    s[i] = next_state

            state1 = self.process_state(numpy.stack(ns))
            reward = torch.tensor(numpy.stack(r), dtype=torch.float32)
            reward_avg.update(reward.mean())
            done = torch.tensor(numpy.stack(d), dtype=torch.float32)

            agent.train_n_env(state0, value, action0, probs0, state1, reward, done)

            state0 = self.process_state(numpy.stack(s))

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Saving data...')
        save_data = {
            're': numpy.array(train_ext_rewards)
        }
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)

    def run_forward_model(self, agent, trial):
        config = self._config
        n_env = config.n_env
        trial = trial + config.shift
        step_limit = int(config.steps * 1e6)
        steps = 0
        bar = ProgressBar(step_limit, max_width=40)

        forward_model = agent.get_motivation_module()

        train_ext_rewards = []
        train_ext_reward = [0] * n_env
        train_int_rewards = []
        train_int_reward = [0] * n_env
        train_fm_errors = []
        train_fm_error = [[] for _ in range(n_env)]
        train_steps = [0] * n_env

        s = [None] * n_env
        ns = [None] * n_env
        r = [None] * n_env
        d = [None] * n_env

        for i in range(n_env):
            s[i] = self._env_list[i].reset()

        state0 = self.process_state(numpy.stack(s))

        while steps < step_limit:
            action0 = agent.get_action(state0)

            for i in range(n_env):
                next_state, reward, done, info = self._env_list[i].step(action0[i].item())
                mask = 1.
                if done:
                    mask = 0.

                train_ext_reward[i] += reward
                train_steps[i] += 1
                ns[i] = next_state
                r[i] = reward
                d[i] = mask

            state1 = self.process_state(numpy.stack(ns))

            fm_error = forward_model.error(state0, action0, state1)
            fm_reward = forward_model.reward(error=fm_error)

            for i in range(n_env):
                train_int_reward[i] += fm_reward[i].item()
                train_fm_error[i].append(fm_error[i].item())

                if d[i] == 0:
                    if steps + train_steps[i] > step_limit:
                        train_steps[i] = step_limit - steps
                    steps += train_steps[i]

                    train_ext_rewards.append([train_steps[i], train_ext_reward[i]])
                    train_int_rewards.append([train_steps[i], train_int_reward[i]])
                    train_fm_errors += train_fm_error[i]
                    bar.numerator = steps

                    print('Run {0:d} step {1:d} training [ext. reward {2:f} c int. reward {3:f} c fm. error {4:f} ps steps {5:d}]'.format(
                        trial, steps, train_ext_reward[i], train_int_reward[i], numpy.array(train_fm_error[i]).mean(), train_steps[i]))
                    print(bar)

                    train_ext_reward[i] = 0
                    train_int_reward[i] = 0
                    train_steps[i] = 0
                    train_fm_error[i].clear()

                    s[i] = self._env_list[i].reset()
                else:
                    s[i] = ns[i]

            reward = torch.tensor(numpy.stack(r), dtype=torch.float32)
            done = torch.tensor(numpy.stack(d), dtype=torch.float32)

            agent.train_n_env(state0, action0, state1, reward, done)

            state0 = self.process_state(numpy.stack(s))

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))
        numpy.save('ppo_{0}_{1}_{2:d}_re'.format(config.name, config.model, trial), numpy.array(train_ext_rewards))
        numpy.save('ppo_{0}_{1}_{2:d}_ri'.format(config.name, config.model, trial), numpy.array(train_int_rewards))
        numpy.save('ppo_{0}_{1}_{2:d}_fme'.format(config.name, config.model, trial), numpy.array(train_fm_errors[:step_limit]))

        print('Saving data...')
        save_data = {
            're': numpy.array(train_ext_rewards),
            'ri': numpy.array(train_int_rewards),
            'fme': numpy.array(train_fm_errors[:step_limit])
        }
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)
