import numpy
import torch
from etaprogress.progress import ProgressBar
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from utils import one_hot_code


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
        agent.load(config.load)

        for i in range(12):
            video_path = 'ppo_{0}_{1}_{2:d}.mp4'.format(config.name, config.model, i)
            video_recorder = VideoRecorder(self._env, video_path, enabled=video_path is not None)
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
        bar = ProgressBar(config.steps * 1e6, max_width=40)
        ext_rewards = []
        rewards = numpy.zeros(100)
        reward_index = 0

        step_limit = config.steps * 1e6
        steps = 0

        while steps < step_limit:
            bar.numerator = steps

            if self._preprocess is None:
                state0 = torch.tensor(self._env.reset(), dtype=torch.float32).to(config.device)
            else:
                state0 = self._preprocess(self._env.reset()).to(config.device)
            done = False
            total_reward = 0
            train_steps = 0

            while not done:
                action0 = agent.get_action(state0)
                next_state, reward, done, info = self._env.step(action0.item())
                total_reward += reward
                train_steps += 1

                if self._preprocess is None:
                    state1 = torch.tensor(next_state, dtype=torch.float32).to(config.device)
                else:
                    state1 = self._preprocess(next_state).to(config.device)

                agent.train(state0, action0, state1, reward, done)
                state0 = state1

            ext_rewards.append(total_reward)
            rewards[reward_index] = total_reward
            reward_index += 1
            steps += train_steps
            if reward_index == 100:
                reward_index = 0

            print('Step {0:d} training [avg. reward {1:f} steps {2:d}]'.format(steps, rewards.mean(), train_steps))
            print(bar)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))
        numpy.save('ppo_{0}_{1}_{2:d}_re'.format(config.name, config.model, trial), numpy.array(ext_rewards))

    def run_icm(self, agent, trial):
        config = self._config
        trial = trial + config.shift
        motivation = agent.get_motivation()

        bar = ProgressBar(config.steps * 1e6, max_width=40)
        ext_rewards = []
        int_rewards = []
        rewards = numpy.zeros(100)
        reward_index = 0

        step_limit = config.steps * 1e6
        steps = 0

        while steps < step_limit:
            bar.numerator = steps

            state0 = torch.tensor(self._env.reset(), dtype=torch.float32).unsqueeze(0).to(config.device)
            done = False
            total_e_reward = 0
            total_i_reward = 0
            train_steps = 0

            while not done:
                action0 = agent.get_action(state0)
                next_state, reward, done, info = self._env.step(action0.item())
                total_e_reward += reward
                train_steps += 1
                state1 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(config.device)
                total_i_reward += motivation.reward(state0, one_hot_code(action0, agent.action_dim(), config.device), state1).item()
                agent.train(state0, action0, state1, reward, done)
                state0 = state1

            ext_rewards.append(total_e_reward)
            int_rewards.append(total_i_reward)
            rewards[reward_index] = total_e_reward
            reward_index += 1
            steps += train_steps
            if reward_index == 100:
                reward_index = 0

            print('Step {0:d} training [avg. reward {1:f} steps {2:d} int. reward {3:f}]'.format(steps, rewards.mean(), train_steps, total_i_reward))
            print(bar)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))
        numpy.save('ppo_{0}_{1}_{2:d}_re'.format(config.name, config.model, trial), numpy.array(ext_rewards))
        numpy.save('ppo_{0}_{1}_{2:d}_ri'.format(config.name, config.model, trial), numpy.array(int_rewards))
