import time
import numpy
import torch
from etaprogress.progress import ProgressBar
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from utils import one_hot_code
from utils.RunningAverage import RunningAverageWindow, StepCounter
from concurrent.futures import ThreadPoolExecutor


class ExperimentNEnvPPO:
    def __init__(self, env_name, env_list, config, input_shape, action_shape):
        self._env_name = env_name
        self._env = env_list[0]
        self._env_list = env_list
        self._config = config
        self._preprocess = None
        self._input_shape = input_shape
        self._action_shape = action_shape

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

    def one_step_baseline(self, params):
        i, trial, agent, action0, train_ext_reward, train_steps, s, ns, r, d, step_counter, train_ext_rewards, reward_avg = params
        next_state, reward, done, info = self._env_list[i].step(action0[i])
        mask = 1
        if done:
            mask = 0

        if info is not None and 'raw_score' in info:
            train_ext_reward['raw'][i] += info['raw_score']
        train_ext_reward['train'][i] += reward
        train_steps[i] += 1
        ns[i] = next_state
        r[i] = reward
        d[i] = mask

        if d[i] == 0:
            if step_counter.steps + train_steps[i] > step_counter.limit:
                train_steps[i] = step_counter.limit - step_counter.steps
            step_counter.update(train_steps[i])

            if info is not None and 'raw_score' in info:
                train_ext_rewards['raw'].append([train_steps[i], train_ext_reward['raw'][i]])
            train_ext_rewards['train'].append([train_steps[i], train_ext_reward['train'][i]])
            reward_avg.update(train_ext_reward['train'][i])

            print('Run {0:d} step {1:d} training [ext. reward {2:f} steps {3:d}] avg. reward {4:f}'.format(trial, step_counter.steps, train_ext_reward['train'][i], train_steps[i], reward_avg.value()))
            step_counter.print()

            if info is not None and 'raw_score' in info:
                train_ext_reward['raw'][i] = 0
            train_ext_reward['train'][i] = 0
            train_steps[i] = 0

            s[i] = self._env_list[i].reset()
        else:
            s[i] = ns[i]

    def one_step_forward_model(self, params):
        i, trial, agent, action0, train_ext_reward, train_int_reward, train_fm_error, \
        train_steps, s, ns, r, d, step_counter, train_ext_rewards, train_int_rewards, train_fm_errors, reward_avg = params

        next_state, reward, done, info = self._env_list[i].step(action0[i])
        mask = 1.
        if done:
            mask = 0.

        if info is not None and 'raw_score' in info:
            train_ext_reward['raw'][i] += info['raw_score']
        train_ext_reward['train'][i] += reward
        train_steps[i] += 1
        ns[i] = next_state
        r[i] = reward
        d[i] = mask

        if d[i] == 0:
            if step_counter.steps + train_steps[i] > step_counter.limit:
                train_steps[i] = step_counter.limit - step_counter.steps
            step_counter.update(train_steps[i])

            if info is not None and 'raw_score' in info:
                train_ext_rewards['raw'].append([train_steps[i], train_ext_reward['raw'][i]])
            train_ext_rewards['train'].append([train_steps[i], train_ext_reward['train'][i]])
            train_int_rewards.append([train_steps[i], train_int_reward[i]])
            train_fm_errors += train_fm_error[i]

            print('Run {0:d} step {1:d} training [ext. reward {2:f} c int. reward {3:f} c fm. error {4:f} ps steps {5:d}]'.format(
                trial, step_counter.steps, train_ext_reward['train'][i], train_int_reward[i], numpy.array(train_fm_error[i]).mean(), train_steps[i]))
            step_counter.print()

            if info is not None and 'raw_score' in info:
                train_ext_reward['raw'][i] = 0
            train_ext_reward['train'][i] = 0
            train_int_reward[i] = 0
            train_steps[i] = 0
            train_fm_error[i].clear()

            s[i] = self._env_list[i].reset()
        else:
            s[i] = ns[i]

    def run_baseline(self, agent, trial):
        config = self._config
        n_env = config.n_env
        trial = trial + config.shift
        step_counter = StepCounter(int(config.steps * 1e6))

        train_ext_rewards = {'raw': [], 'train': []}
        train_ext_reward = {'raw': [0] * n_env, 'train': [0] * n_env}
        train_steps = [0] * n_env
        reward_avg = RunningAverageWindow(100)
        time_avg = RunningAverageWindow(100)

        s = numpy.zeros((n_env,) + self._input_shape, dtype=numpy.float32)
        a = [None] * n_env
        ns = numpy.zeros((n_env,) + self._input_shape, dtype=numpy.float32)
        r = numpy.zeros((n_env, 1), dtype=numpy.float32)
        d = numpy.zeros((n_env, 1), dtype=numpy.float32)

        inputs = []
        for i in range(n_env):
            inputs.append((i, trial, agent, a, train_ext_reward, train_steps, s, ns, r, d, step_counter, train_ext_rewards, reward_avg))

        for i in range(n_env):
            s[i] = self._env_list[i].reset()

        state0 = self.process_state(s)

        while step_counter.running():
            value, action0, probs0 = agent.get_action(state0)

            for i in range(n_env):
                a[i] = agent.convert_action(action0[i])

            start = time.time()
            with ThreadPoolExecutor(max_workers=config.num_threads) as executor:
                executor.map(self.one_step_baseline, inputs)

            end = time.time()
            time_avg.update(end - start)
            # print('Duration {0:.3f}s'.format(time_avg.value()))

            state1 = self.process_state(ns)
            reward = torch.tensor(r, dtype=torch.float32)
            done = torch.tensor(d, dtype=torch.float32)

            agent.train(state0, value, action0, probs0, state1, reward, done)

            state0 = self.process_state(s)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Saving data...')
        save_data = {
            're': numpy.array(train_ext_rewards['train']),
            're_raw': numpy.array(train_ext_rewards['raw'])
        }
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)

    def run_forward_model(self, agent, trial):
        config = self._config
        n_env = config.n_env
        trial = trial + config.shift
        step_counter = StepCounter(int(config.steps * 1e6))
        reward_avg = RunningAverageWindow(100)
        time_avg = RunningAverageWindow(100)

        train_ext_rewards = {'raw': [], 'train': []}
        train_ext_reward = {'raw': [0] * n_env, 'train': [0] * n_env}
        train_int_rewards = []
        train_int_reward = [0] * n_env
        train_fm_errors = []
        train_fm_error = [[] for _ in range(n_env)]
        train_steps = [0] * n_env

        s = numpy.zeros((n_env,) + self._input_shape, dtype=numpy.float32)
        a = [None] * n_env
        ns = numpy.zeros((n_env,) + self._input_shape, dtype=numpy.float32)
        r = numpy.zeros((n_env, 1), dtype=numpy.float32)
        d = numpy.zeros((n_env, 1), dtype=numpy.float32)

        inputs = []
        for i in range(n_env):
            inputs.append(
                (i, trial, agent, a, train_ext_reward, train_int_reward, train_fm_error, train_steps, s, ns, r, d, step_counter, train_ext_rewards, train_int_rewards, train_fm_errors, reward_avg))

        for i in range(n_env):
            s[i] = self._env_list[i].reset()

        state0 = self.process_state(s)

        while step_counter.running():
            value, action0, probs0 = agent.get_action(state0)

            for i in range(n_env):
                a[i] = agent.convert_action(action0[i])

            # start = time.time()
            with ThreadPoolExecutor(max_workers=config.num_threads) as executor:
                executor.map(self.one_step_forward_model, inputs)

            # end = time.time()
            # time_avg.update(end - start)
            # print('Duration {0:.3f}s'.format(time_avg.value()))

            state1 = self.process_state(ns)

            fm_error = agent.motivation.error(state0, action0, state1)
            fm_reward = agent.motivation.reward(error=fm_error)

            for i in range(n_env):
                train_int_reward[i] += fm_reward[i].item()
                train_fm_error[i].append(fm_error[i].item())

            reward = torch.stack([torch.tensor(r, dtype=torch.float32), fm_reward.cpu()], dim=1).squeeze(-1)
            done = torch.tensor(d, dtype=torch.float32)

            agent.train(state0, value, action0, probs0, state1, reward, done)

            state0 = self.process_state(s)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))

        print('Saving data...')
        save_data = {
            're': numpy.array(train_ext_rewards['train']),
            're_raw': numpy.array(train_ext_rewards['raw']),
            'ri': numpy.array(train_int_rewards),
            'fme': numpy.array(train_fm_errors[:step_counter.limit])
        }
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)
