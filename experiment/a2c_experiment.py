from collections import deque

import numpy
import torch
from etaprogress.progress import ProgressBar
from gym.wrappers.monitoring.video_recorder import VideoRecorder


class ExperimentA2C:
    def __init__(self, env_name, env, config):
        self._env_name = env_name
        self._env = env
        self._config = config

    def run_baseline(self, agent, trial):
        config = self._config

        if config.load:
            agent.load(config.load)

            for i in range(5):
                video_path = 'a2c_{0}_{1}_{2:d}.mp4'.format(config.name, config.model, i)
                video_recorder = VideoRecorder(self._env, video_path, enabled=video_path is not None)
                state0 = torch.tensor(self._env.reset(), dtype=torch.float32).unsqueeze(0).to(config.device)
                done = False

                while not done:
                    self._env.render()
                    video_recorder.capture_frame()
                    action0, _, _ = agent.get_action(state0)
                    next_state, reward, done, info = self._env.step(action0)
                    state0 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(config.device)
                video_recorder.close()
        else:
            bar = ProgressBar(config.steps * 1e6, max_width=40)
            ext_rewards = []
            rewards = numpy.zeros(100)
            reward_index = 0

            step_limit = config.steps * 1e6
            steps = 0

            while steps < step_limit:
                bar.numerator = steps

                state0 = torch.tensor(self._env.reset(), dtype=torch.float32).unsqueeze(0).to(config.device)
                done = False
                total_reward = 0
                train_steps = 0

                while not done:
                    action0, prob, log_prob = agent.get_action(state0)
                    next_state, reward, done, info = self._env.step(action0)
                    total_reward += reward
                    train_steps += 1
                    agent.train(state0, prob, log_prob, reward, done)
                    state0 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(config.device)

                ext_rewards.append(total_reward)
                rewards[reward_index] = total_reward
                reward_index += 1
                steps += train_steps
                if reward_index == 100:
                    reward_index = 0

                print('Step {0:d} training [avg. reward {1:f} steps {2:d}]'.format(steps, rewards.mean(), train_steps))
                print(bar)

            agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))
            numpy.save('a2c_{0}_{1}_{2:d}_re'.format(config.name, config.model, trial), numpy.array(ext_rewards))
