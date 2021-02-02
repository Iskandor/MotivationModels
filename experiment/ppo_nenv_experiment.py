import time

import numpy
import torch
from etaprogress.progress import ProgressBar
from gym.wrappers.monitoring.video_recorder import VideoRecorder


class ExperimentNEnvPPO:
    def __init__(self, env_name, env_list, config):
        self._env_name = env_name
        self._env = env_list[0]
        self._env_list = env_list
        self._config = config
        self._preprocess = None

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
                mask = 1
                if done:
                    mask = 0

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

                    print('Run {0:d} step {1:d} training [ext. reward {2:f} steps {3:d}]'.format(trial, steps, train_ext_reward[i], train_steps[i]))
                    print(bar)

                    train_ext_reward[i] = 0
                    train_steps[i] = 0

                    s[i] = self._env_list[i].reset()
                else:
                    s[i] = next_state

            state1 = self.process_state(numpy.stack(ns))
            reward = torch.tensor(numpy.stack(r), dtype=torch.float32)
            done = torch.tensor(numpy.stack(d), dtype=torch.float32)

            agent.train_n_env(state0, action0, state1, reward, done)

            state0 = self.process_state(numpy.stack(s))

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))
        numpy.save('ppo_{0}_{1}_{2:d}_re'.format(config.name, config.model, trial), numpy.array(train_ext_rewards))

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

                    print('Run {0:d} step {1:d} training [ext. reward {2:f} int. reward {3:f} steps {4:d}]'.format(
                        trial, steps, train_ext_reward[i], train_int_reward[i], train_steps[i]))
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
