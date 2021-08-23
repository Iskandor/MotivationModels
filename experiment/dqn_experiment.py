import numpy
import torch
from etaprogress.progress import ProgressBar

from exploration.DiscreteExploration import DiscreteExploration
from utils.RunningAverage import RunningAverageWindow


class ExperimentDQN:
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

    def run_baseline(self, agent, trial):
        config = self._config
        trial = trial + config.shift
        step_limit = int(config.steps * 1e6)
        steps = 0

        steps_per_episode = []
        train_ext_rewards = []
        reward_avg = RunningAverageWindow(100)

        bar = ProgressBar(config.steps * 1e6, max_width=40)
        exploration = DiscreteExploration(config.epsilon, 0, config.steps * config.exploration_time * 1e6)

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
                action0 = exploration.explore(agent.convert_action(agent.get_action(state0)), self._env)
                next_state, reward, done, _ = self._env.step(action0)
                reward = self.transform_reward(reward)
                train_ext_reward += reward

                if self._preprocess is None:
                    state1 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                else:
                    state1 = self._preprocess(next_state)

                reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
                mask = torch.tensor([done], dtype=torch.float32).unsqueeze(0)
                action0 = torch.tensor([action0], dtype=torch.int64).unsqueeze(0)

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

            print('Run {0:d} step {1:d} epsilon {2:f} training [ext. reward {3:f} steps {4:d}] avg. ext. reward {5:f}'.format(
                trial, steps, exploration.epsilon, train_ext_reward, train_steps, reward_avg.value().item()))
            print(bar)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Saving data...')
        save_data = {
            'steps': numpy.array(steps_per_episode),
            're': numpy.array(train_ext_rewards)
        }
        numpy.save('dqn_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)