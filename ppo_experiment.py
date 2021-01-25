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

        while steps < step_limit:
            bar.numerator = steps

            if self._preprocess is None:
                state0 = torch.tensor(self._env.reset(), dtype=torch.float32).to(config.device)
            else:
                state0 = self._preprocess(self._env.reset()).to(config.device)
            done = False
            train_ext_reward = 0
            train_steps = 0

            while not done:
                action0 = agent.get_action(state0)
                next_state, reward, done, info = self._env.step(action0.item())
                train_ext_reward += reward
                train_steps += 1

                if self._preprocess is None:
                    state1 = torch.tensor(next_state, dtype=torch.float32).to(config.device)
                else:
                    state1 = self._preprocess(next_state).to(config.device)

                agent.train(state0, action0, state1, reward, done)
                state0 = state1

            if steps + train_steps > step_limit:
                train_steps = step_limit - steps
            steps += train_steps
            bar.numerator = steps

            train_ext_rewards.append([train_steps, train_ext_reward])

            print('Run {0:d} step {1:d} training [ext. reward {2:f} steps {3:d}]'.format(trial, steps, train_ext_reward, train_steps))
            print(bar)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))
        numpy.save('ppo_{0}_{1}_{2:d}_re'.format(config.name, config.model, trial), numpy.array(train_ext_rewards))
