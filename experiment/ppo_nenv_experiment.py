import time
import numpy
import torch
from etaprogress.progress import ProgressBar
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from utils import one_hot_code
from utils.RunningAverage import RunningAverageWindow, StepCounter
from concurrent.futures import ThreadPoolExecutor


class ExperimentNEnvPPO:
    def __init__(self, env_name, env, config):
        self._env_name = env_name
        self._env = env
        self._config = config
        self._preprocess = None

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
        step_counter = StepCounter(int(config.steps * 1e6))

        steps_per_episode = []
        train_ext_rewards = []
        train_vars = []
        train_ext_reward = numpy.zeros((n_env, 1), dtype=numpy.float32)
        train_steps = numpy.zeros((n_env, 1), dtype=numpy.int32)
        # train_var = numpy.zeros((n_env, self._env.action_space.shape[0]), dtype=numpy.float32)
        reward_avg = RunningAverageWindow(100)
        # time_avg = RunningAverageWindow(100)

        s = numpy.zeros((n_env,) + self._env.observation_space.shape, dtype=numpy.float32)
        for i in range(n_env):
            s[i] = self._env.reset(i)

        state0 = self.process_state(s)

        while step_counter.running():
            with torch.no_grad():
                value, action0, probs0 = agent.get_action(state0)

            # start = time.time()
            next_state, reward, done, info = self._env.step(agent.convert_action(action0.cpu()))
            # end = time.time()
            # time_avg.update(end - start)
            # print('Duration {0:.3f}s'.format(time_avg.value()))

            train_steps += 1
            if info is not None and 'raw_score' in info:
                train_ext_reward += numpy.expand_dims(info['raw_score'], axis=1)
            else:
                train_ext_reward += reward
            # var = probs0[:, self._env.action_space.shape[0]:]
            # train_var += var.cpu().numpy()

            env_indices = numpy.nonzero(numpy.squeeze(done, axis=1))[0]

            for i in env_indices:
                if step_counter.steps + train_steps[i] > step_counter.limit:
                    train_steps[i] = step_counter.limit - step_counter.steps
                step_counter.update(train_steps[i].item())

                steps_per_episode.append(train_steps[i].item())
                train_ext_rewards.append(train_ext_reward[i].item())
                # train_vars.append(train_var[i] / train_steps[i].item())
                reward_avg.update(train_ext_reward[i].item())

                print('Run {0:d} step {1:d} training [ext. reward {2:f} steps {3:d} avg. reward {4:f}]'.format(trial, step_counter.steps, train_ext_reward[i].item(), train_steps[i].item(),
                                                                                                               reward_avg.value().item()))
                step_counter.print()

                train_ext_reward[i] = 0
                train_steps[i] = 0
                # train_var[i].fill(0)

                next_state[i] = self._env.reset(i)

            state1 = self.process_state(next_state)
            reward = torch.tensor(reward, dtype=torch.float32)
            done = torch.tensor(done, dtype=torch.float32)

            agent.train(state0, value, action0, probs0, state1, reward, done)

            state0 = state1

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Saving data...')
        save_data = {
            'steps': numpy.array(steps_per_episode),
            're': numpy.array(train_ext_rewards),
            'var': numpy.array(train_vars),
            'diff_mean': numpy.array(agent.algorithm.diff_mean),
            'diff_max': numpy.array(agent.algorithm.diff_max),
        }
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)

    def run_rnd_model(self, agent, trial):
        config = self._config
        n_env = config.n_env
        trial = trial + config.shift
        step_counter = StepCounter(int(config.steps * 1e6))

        steps_per_episode = []
        train_ext_rewards = []
        train_ext_reward = numpy.zeros((n_env, 1), dtype=numpy.float32)
        train_int_rewards = []
        train_int_reward = numpy.zeros((n_env, 1), dtype=numpy.float32)
        train_error = [[] for _ in range(n_env)]
        train_errors = numpy.array([], dtype=numpy.int32)
        train_steps = numpy.zeros((n_env, 1), dtype=numpy.int32)
        reward_avg = RunningAverageWindow(100)
        # time_avg = RunningAverageWindow(100)

        s = numpy.zeros((n_env,) + self._env.observation_space.shape, dtype=numpy.float32)
        for i in range(n_env):
            s[i] = self._env.reset(i)

        state0 = self.process_state(s)

        while step_counter.running():
            with torch.no_grad():
                value, action0, probs0 = agent.get_action(state0)

            # start = time.time()
            next_state, reward, done, info = self._env.step(agent.convert_action(action0.cpu()))
            # end = time.time()
            # time_avg.update(end - start)
            # print('Duration {0:.3f}s'.format(time_avg.value()))

            ext_reward = torch.tensor(reward, dtype=torch.float32)
            int_reward = agent.motivation.reward(state0).cpu()

            error = agent.motivation.error(state0).cpu().tolist()
            for i in range(n_env):
                train_error[i].append(error[i])

            train_steps += 1
            train_ext_reward += ext_reward.numpy()
            train_int_reward += int_reward.numpy()

            env_indices = numpy.nonzero(numpy.squeeze(done, axis=1))[0]

            for i in env_indices:
                if step_counter.steps + train_steps[i] > step_counter.limit:
                    train_steps[i] = step_counter.limit - step_counter.steps
                step_counter.update(train_steps[i].item())

                steps_per_episode.append(train_steps[i].item())
                train_ext_rewards.append(train_ext_reward[i].item())
                train_int_rewards.append(train_int_reward[i].item())
                train_errors = numpy.concatenate([train_errors, numpy.array(train_error[i])])
                reward_avg.update(train_ext_reward[i].item())

                if train_steps[i].item() > 0:
                    print('Run {0:d} step {1:d} training [ext. reward {2:f} int. reward {3:f} steps {4:d} ({5:f})  mean reward {6:f}]'.format(
                        trial, step_counter.steps, train_ext_reward[i].item(), train_int_reward[i].item(), train_steps[i].item(), train_int_reward[i].item() / train_steps[i].item(),
                        reward_avg.value().item()))
                step_counter.print()

                train_ext_reward[i] = 0
                train_int_reward[i] = 0
                train_steps[i] = 0
                del train_error[i][:]

                next_state[i] = self._env.reset(i)

            state1 = self.process_state(next_state)

            reward = torch.cat([ext_reward, int_reward], dim=1)
            done = torch.tensor(done, dtype=torch.float32)

            agent.train(state0, value, action0, probs0, state1, reward, done)

            state0 = state1

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Saving data...')
        save_data = {
            'steps': numpy.array(steps_per_episode),
            're': numpy.array(train_ext_rewards),
            'ri': numpy.array(train_int_rewards),
            'fme': numpy.array(train_errors[:step_counter.limit])
        }
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)

    def run_qrnd_model(self, agent, trial):
        config = self._config
        n_env = config.n_env
        trial = trial + config.shift
        step_counter = StepCounter(int(config.steps * 1e6))

        steps_per_episode = []
        train_ext_rewards = []
        train_ext_reward = numpy.zeros((n_env, 1), dtype=numpy.float32)
        train_int_rewards = []
        train_int_reward = numpy.zeros((n_env, 1), dtype=numpy.float32)
        train_error = [[] for _ in range(n_env)]
        train_errors = numpy.array([], dtype=numpy.int32)
        train_steps = numpy.zeros((n_env, 1), dtype=numpy.int32)
        reward_avg = RunningAverageWindow(100)
        # time_avg = RunningAverageWindow(100)

        s = numpy.zeros((n_env,) + self._env.observation_space.shape, dtype=numpy.float32)
        for i in range(n_env):
            s[i] = self._env.reset(i)

        state0 = self.process_state(s)

        while step_counter.running():
            with torch.no_grad():
                value, action0, probs0 = agent.get_action(state0)

            # start = time.time()
            next_state, reward, done, info = self._env.step(agent.convert_action(action0.cpu()))
            # end = time.time()
            # time_avg.update(end - start)
            # print('Duration {0:.3f}s'.format(time_avg.value()))

            ext_reward = torch.tensor(reward, dtype=torch.float32)
            int_reward = agent.motivation.reward(state0, action0).cpu()

            error = agent.motivation.error(state0, action0).cpu().tolist()
            for i in range(n_env):
                train_error[i].append(error[i])

            train_steps += 1
            train_ext_reward += ext_reward.numpy()
            train_int_reward += int_reward.numpy()

            env_indices = numpy.nonzero(numpy.squeeze(done, axis=1))[0]

            for i in env_indices:
                if step_counter.steps + train_steps[i] > step_counter.limit:
                    train_steps[i] = step_counter.limit - step_counter.steps
                step_counter.update(train_steps[i].item())

                steps_per_episode.append(train_steps[i].item())
                train_ext_rewards.append(train_ext_reward[i].item())
                train_int_rewards.append(train_int_reward[i].item())
                train_errors = numpy.concatenate([train_errors, numpy.array(train_error[i])])
                reward_avg.update(train_ext_reward[i].item())

                if train_steps[i].item() > 0:
                    print('Run {0:d} step {1:d} training [ext. reward {2:f} int. reward {3:f} steps {4:d} ({5:f})  mean reward {6:f}]'.format(
                        trial, step_counter.steps, train_ext_reward[i].item(), train_int_reward[i].item(), train_steps[i].item(), train_int_reward[i].item() / train_steps[i].item(),
                        reward_avg.value().item()))
                step_counter.print()

                train_ext_reward[i] = 0
                train_int_reward[i] = 0
                train_steps[i] = 0
                del train_error[i][:]

                next_state[i] = self._env.reset(i)

            state1 = self.process_state(next_state)

            reward = torch.cat([ext_reward, int_reward], dim=1)
            done = torch.tensor(done, dtype=torch.float32)

            agent.train(state0, value, action0, probs0, state1, reward, done)

            state0 = state1

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Saving data...')
        save_data = {
            'steps': numpy.array(steps_per_episode),
            're': numpy.array(train_ext_rewards),
            'ri': numpy.array(train_int_rewards),
            'fme': numpy.array(train_errors[:step_counter.limit])
        }
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)

    def run_dop_model(self, agent, trial):
        config = self._config
        n_env = config.n_env
        trial = trial + config.shift
        step_counter = StepCounter(int(config.steps * 1e6))

        steps_per_episode = []
        train_ext_rewards = []
        train_ext_reward = numpy.zeros((n_env, 1), dtype=numpy.float32)
        train_int_rewards = []
        train_int_reward = numpy.zeros((n_env, 1), dtype=numpy.float32)
        train_error = [[] for _ in range(n_env)]
        train_errors = numpy.array([], dtype=numpy.int32)
        train_steps = numpy.zeros((n_env, 1), dtype=numpy.int32)
        head_index_density = numpy.zeros((n_env, config.dop_heads))
        reward_avg = RunningAverageWindow(100)
        # time_avg = RunningAverageWindow(100)

        s = numpy.zeros((n_env,) + self._env.observation_space.shape, dtype=numpy.float32)
        for i in range(n_env):
            s[i] = self._env.reset(i)

        state0 = self.process_state(s)

        while step_counter.running():
            with torch.no_grad():
                value, action0, probs0 = agent.get_action(state0)
                head_index = agent.get_head_index().unsqueeze(-1)

            next_state, reward, done, info = self._env.step(agent.convert_action(action0.cpu()))

            head_index = one_hot_code(head_index, config.dop_heads).numpy()
            head_index_density += head_index
            ext_reward = torch.tensor(reward, dtype=torch.float32)
            int_reward = agent.motivation.reward(state0, action0).cpu()

            error = agent.motivation.error(state0, action0).cpu().tolist()
            for i in range(n_env):
                train_error[i].append(error[i])

            train_steps += 1
            train_ext_reward += ext_reward.numpy()
            train_int_reward += int_reward.numpy()

            env_indices = numpy.nonzero(numpy.squeeze(done, axis=1))[0]

            for i in env_indices:
                if step_counter.steps + train_steps[i] > step_counter.limit:
                    train_steps[i] = step_counter.limit - step_counter.steps
                step_counter.update(train_steps[i].item())

                steps_per_episode.append(train_steps[i].item())
                train_ext_rewards.append(train_ext_reward[i].item())
                train_int_rewards.append(train_int_reward[i].item())
                train_errors = numpy.concatenate([train_errors, numpy.array(train_error[i])])
                reward_avg.update(train_ext_reward[i].item())

                if train_steps[i].item() > 0:
                    print('Run {0:d} step {1:d} training [ext. reward {2:f} int. reward {3:f} steps {4:d} ({5:f})  mean reward {6:f} density {7:s}]'.format(
                        trial, step_counter.steps, train_ext_reward[i].item(), train_int_reward[i].item(), train_steps[i].item(), train_int_reward[i].item() / train_steps[i].item(),
                        reward_avg.value().item(), numpy.array2string(head_index_density[i])))
                step_counter.print()

                train_ext_reward[i] = 0
                train_int_reward[i] = 0
                train_steps[i] = 0
                head_index_density[i].fill(0)
                del train_error[i][:]

                next_state[i] = self._env.reset(i)

            state1 = self.process_state(next_state)

            reward = torch.tensor(ext_reward, dtype=torch.float32)
            done = torch.tensor(done, dtype=torch.float32)

            agent.train(state0, value, action0, probs0, state1, reward, done)

            state0 = state1

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Saving data...')
        save_data = {
            'steps': numpy.array(steps_per_episode),
            're': numpy.array(train_ext_rewards),
            'ri': numpy.array(train_int_rewards),
            'fme': numpy.array(train_errors[:step_counter.limit])
        }
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)
