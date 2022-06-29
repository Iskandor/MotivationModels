import numpy
import torch
from etaprogress.progress import ProgressBar
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from analytic.CNDAnalytic import CNDAnalytic
from analytic.RNDAnalytic import RNDAnalytic
from utils import one_hot_code
from utils.RunningAverage import RunningAverageWindow, StepCounter, RunningStats
from concurrent.futures import ThreadPoolExecutor

from utils.TimeEstimator import PPOTimeEstimator


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

        for i in range(1):
            video_path = 'ppo_{0}_{1}_{2:d}.mp4'.format(config.name, config.model, i)
            video_recorder = VideoRecorder(self._env, video_path, enabled=video_path is not None)
            state0 = torch.tensor(self._env.reset(), dtype=torch.float32).unsqueeze(0).to(config.device)
            done = False

            while not done:
                self._env.render()
                video_recorder.capture_frame()
                # features0 = agent.get_features(state0)
                _, _, action0, probs0 = agent.get_action(state0)
                # actor_state, value, action0, probs0, head_value, head_action, head_probs, all_values, all_action, all_probs = agent.get_action(state0)
                # action0 = probs0.argmax(dim=1)
                next_state, reward, done, info = self._env.step(agent.convert_action(action0.cpu()))
                state0 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(config.device)
            video_recorder.close()

    def run_baseline(self, agent, trial):
        config = self._config
        n_env = config.n_env
        trial = trial + config.shift
        step_counter = StepCounter(int(config.steps * 1e6))

        steps_per_episode = []
        train_ext_rewards = []
        train_ext_reward = numpy.zeros((n_env, 1), dtype=numpy.float32)
        train_scores = []
        train_score = numpy.zeros((n_env, 1), dtype=numpy.float32)
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
            train_ext_reward += reward
            if info is not None and 'raw_score' in info:
                score = numpy.expand_dims(info['raw_score'], axis=1)
                train_score += score
            # var = probs0[:, self._env.action_space.shape[0]:]
            # train_var += var.cpu().numpy()

            env_indices = numpy.nonzero(numpy.squeeze(done, axis=1))[0]

            for i in env_indices:
                if step_counter.steps + train_steps[i] > step_counter.limit:
                    train_steps[i] = step_counter.limit - step_counter.steps
                step_counter.update(train_steps[i].item())

                steps_per_episode.append(train_steps[i].item())
                train_ext_rewards.append(train_ext_reward[i].item())
                train_scores.append(train_score[i].item())
                # train_vars.append(train_var[i] / train_steps[i].item())
                reward_avg.update(train_ext_reward[i].item())

                print('Run {0:d} step {1:d} training [ext. reward {2:f} steps {3:d} avg. reward {4:f} score {5:f}]'.format(trial, step_counter.steps, train_ext_reward[i].item(), train_steps[i].item(),
                                                                                                                           reward_avg.value().item(), train_score[i].item()))
                step_counter.print()

                train_ext_reward[i] = 0
                train_score[i] = 0
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
            'score': numpy.array(train_scores),
            're': numpy.array(train_ext_rewards)
        }
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)

    def run_rnd_model(self, agent, trial):
        config = self._config
        n_env = config.n_env
        trial = trial + config.shift
        step_counter = StepCounter(int(config.steps * 1e6))

        analytic = RNDAnalytic()
        analytic.init(n_env, ext_reward=(1,), score=(1,), int_reward=(1,), error=(1,), ext_value=(1,), int_value=(1,))

        reward_avg = RunningAverageWindow(100)
        time_estimator = PPOTimeEstimator(step_counter.limit)

        s = numpy.zeros((n_env,) + self._env.observation_space.shape, dtype=numpy.float32)
        for i in range(n_env):
            s[i] = self._env.reset(i)

        state0 = self.process_state(s)

        while step_counter.running():
            agent.motivation.update_state_average(state0)
            with torch.no_grad():
                value, action0, probs0 = agent.get_action(state0)
            next_state, reward, done, info = self._env.step(agent.convert_action(action0.cpu()))

            ext_reward = torch.tensor(reward, dtype=torch.float32)
            int_reward = agent.motivation.reward(state0).cpu().clip(0.0, 1.0)

            if info is not None and 'raw_score' in info:
                score = torch.tensor(info['raw_score']).unsqueeze(-1)
                analytic.update(score=score)

            error = agent.motivation.error(state0).cpu()
            analytic.update(ext_reward=ext_reward,
                            int_reward=int_reward,
                            ext_value=value[:, 0].unsqueeze(-1).cpu(),
                            int_value=value[:, 1].unsqueeze(-1).cpu(),
                            error=error)

            env_indices = numpy.nonzero(numpy.squeeze(done, axis=1))[0]
            stats = analytic.reset(env_indices)
            step_counter.update(n_env)

            for i, index in enumerate(env_indices):
                reward_avg.update(stats['ext_reward'].sum[i].item())

                print('Run {0:d} step {1:d}/{2:d} training [ext. reward {3:f} int. reward (max={4:f} mean={5:f} std={6:f}) steps {7:d}  mean reward {8:f} score {9:f})]'.format(
                    trial, step_counter.steps, step_counter.limit, stats['ext_reward'].sum[i].item(), stats['int_reward'].max[i].item(), stats['int_reward'].mean[i].item(), stats['int_reward'].std[i].item(),
                    int(stats['ext_reward'].step[i].item()), reward_avg.value().item(), stats['score'].sum[i].item()))
                print(time_estimator)
                next_state[i] = self._env.reset(index)

            state1 = self.process_state(next_state)

            reward = torch.cat([ext_reward, int_reward], dim=1)
            done = torch.tensor(1 - done, dtype=torch.float32)
            analytic.end_step()

            agent.train(state0, value, action0, probs0, state1, reward, done)

            state0 = state1
            time_estimator.update(n_env)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(config.name, config.model, trial))

        print('Saving data...')
        analytic.reset(numpy.array(range(n_env)))
        save_data = analytic.finalize()
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)
        analytic.clear()

    def run_qrnd_model(self, agent, trial):
        config = self._config
        n_env = config.n_env
        trial = trial + config.shift
        step_counter = StepCounter(int(config.steps * 1e6))

        analytic = RNDAnalytic()
        analytic.init(n_env, ext_reward=(1,), score=(1,), int_reward=(1,), error=(1,), ext_value=(1,), int_value=(1,))

        reward_avg = RunningAverageWindow(100)
        time_estimator = PPOTimeEstimator(step_counter.limit)

        s = numpy.zeros((n_env,) + self._env.observation_space.shape, dtype=numpy.float32)
        for i in range(n_env):
            s[i] = self._env.reset(i)

        state0 = self.process_state(s)

        while step_counter.running():

            with torch.no_grad():
                value, action0, probs0 = agent.get_action(state0)
            agent.motivation.update_state_average(state0, action0)
            next_state, reward, done, info = self._env.step(agent.convert_action(action0.cpu()))

            error = agent.motivation.error(state0, action0).cpu()
            ext_reward = torch.tensor(reward, dtype=torch.float32)
            int_reward = agent.motivation.reward(error).cpu().clip(0.0, 1.0)

            if info is not None and 'raw_score' in info:
                score = torch.tensor(info['raw_score']).unsqueeze(-1)
                analytic.update(score=score)

            analytic.update(ext_reward=ext_reward,
                            int_reward=int_reward,
                            ext_value=value[:, 0].unsqueeze(-1).cpu(),
                            int_value=value[:, 1].unsqueeze(-1).cpu(),
                            error=error)

            env_indices = numpy.nonzero(numpy.squeeze(done, axis=1))[0]
            stats = analytic.reset(env_indices)
            step_counter.update(n_env)

            for i, index in enumerate(env_indices):
                reward_avg.update(stats['ext_reward'].sum[i].item())

                print('Run {0:d} step {1:d}/{2:d} training [ext. reward {3:f} int. reward (max={4:f} mean={5:f} std={6:f}) steps {7:d}  mean reward {8:f} score {9:f})]'.format(
                    trial, step_counter.steps, step_counter.limit, stats['ext_reward'].sum[i].item(), stats['int_reward'].max[i].item(), stats['int_reward'].mean[i].item(), stats['int_reward'].std[i].item(),
                    int(stats['ext_reward'].step[i].item()), reward_avg.value().item(), stats['score'].sum[i].item()))
                print(time_estimator)

                next_state[i] = self._env.reset(index)

            state1 = self.process_state(next_state)

            reward = torch.cat([ext_reward, int_reward], dim=1)
            done = torch.tensor(1 - done, dtype=torch.float32)
            analytic.end_step()

            agent.train(state0, value, action0, probs0, state1, reward, done)

            state0 = state1
            time_estimator.update(n_env)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(config.name, config.model, trial))

        print('Saving data...')
        analytic.reset(numpy.array(range(n_env)))
        save_data = analytic.finalize()
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)
        analytic.clear()

    def run_sr_rnd_model(self, agent, trial):
        config = self._config
        n_env = config.n_env
        trial = trial + config.shift
        step_counter = StepCounter(int(config.steps * 1e6))

        steps_per_episode = []
        train_ext_rewards = []
        train_ext_reward = numpy.zeros((n_env, 1), dtype=numpy.float32)
        train_scores = []
        train_score = numpy.zeros((n_env, 1), dtype=numpy.float32)
        train_int_rewards = []
        train_int_reward = numpy.zeros((n_env, 1), dtype=numpy.float32)
        train_errors = []
        train_error = numpy.zeros((n_env, 1), dtype=numpy.float32)
        train_steps = numpy.zeros((n_env, 1), dtype=numpy.int32)
        reward_avg = RunningAverageWindow(100)

        s = numpy.zeros((n_env,) + self._env.observation_space.shape, dtype=numpy.float32)
        for i in range(n_env):
            s[i] = self._env.reset(i)

        state0 = self.process_state(s)

        while step_counter.running():
            agent.motivation.update_state_average(state0)
            with torch.no_grad():
                features0 = agent.get_features(state0)
                value, action0, probs0 = agent.get_action(features0)
            next_state, reward, done, info = self._env.step(agent.convert_action(action0.cpu()))

            ext_reward = torch.tensor(reward, dtype=torch.float32)
            int_reward = agent.motivation.reward(state0).cpu().clip(0.0, 1.0)

            if info is not None and 'raw_score' in info:
                score = numpy.expand_dims(info['raw_score'], axis=1)
                train_score += score

            error = agent.motivation.error(state0).cpu()
            train_steps += 1
            train_ext_reward += ext_reward.numpy()
            train_int_reward += int_reward.numpy()
            train_error += error.numpy()

            env_indices = numpy.nonzero(numpy.squeeze(done, axis=1))[0]

            for i in env_indices:
                if step_counter.steps + train_steps[i] > step_counter.limit:
                    train_steps[i] = step_counter.limit - step_counter.steps
                step_counter.update(train_steps[i].item())

                steps_per_episode.append(train_steps[i].item())
                train_ext_rewards.append(train_ext_reward[i].item())
                train_int_rewards.append(train_int_reward[i].item())
                train_errors.append(train_error[i].item())
                train_scores.append(train_score[i].item())
                reward_avg.update(train_ext_reward[i].item())

                if train_steps[i].item() > 0:
                    print('Run {0:d} step {1:d} training [ext. reward {2:f} int. reward {3:f} steps {4:d} ({5:f})  mean reward {6:f} score {7:f}]'.format(
                        trial, step_counter.steps, train_ext_reward[i].item(), train_int_reward[i].item(), train_steps[i].item(), train_int_reward[i].item() / train_steps[i].item(),
                        reward_avg.value().item(), train_score[i].item()))
                step_counter.print()

                train_ext_reward[i] = 0
                train_int_reward[i] = 0
                train_score[i] = 0
                train_steps[i] = 0
                train_error[i] = 0

                next_state[i] = self._env.reset(i)

            state1 = self.process_state(next_state)
            features1 = agent.get_features(state1)

            reward = torch.cat([ext_reward, int_reward], dim=1)
            done = torch.tensor(1 - done, dtype=torch.float32)

            agent.train(state0, features0, value, action0, probs0, state1, features1, reward, done)

            state0 = state1

        agent.save('./models/{0:s}_{1}_{2:d}'.format(config.name, config.model, trial))

        print('Saving data...')
        save_data = {
            'steps': numpy.array(steps_per_episode),
            'score': numpy.array(train_scores),
            're': numpy.array(train_ext_rewards),
            'ri': numpy.array(train_int_rewards),
            'error': numpy.array(train_errors)
        }
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)

    def run_cnd_model(self, agent, trial):
        config = self._config
        n_env = config.n_env
        trial = trial + config.shift
        step_counter = StepCounter(int(config.steps * 1e6))

        analytic = CNDAnalytic()
        analytic.init(n_env, ext_reward=(1,), score=(1,), int_reward=(1,), error=(1,), feature_space=(1,), state_space=(1,), ext_value=(1,), int_value=(1,))

        reward_avg = RunningAverageWindow(100)
        time_estimator = PPOTimeEstimator(step_counter.limit)

        s = numpy.zeros((n_env,) + self._env.observation_space.shape, dtype=numpy.float32)
        for i in range(n_env):
            s[i] = self._env.reset(i)

        state0 = self.process_state(s)

        while step_counter.running():
            agent.motivation.update_state_average(state0)
            with torch.no_grad():
                features, value, action0, probs0 = agent.get_action(state0)
            next_state, reward, done, info = self._env.step(agent.convert_action(action0.cpu()))

            ext_reward = torch.tensor(reward, dtype=torch.float32)
            int_reward = agent.motivation.reward(state0).cpu().clip(0.0, 1.0)

            if info is not None and 'raw_score' in info:
                score = torch.tensor(info['raw_score']).unsqueeze(-1)
                analytic.update(score=score)

            error = agent.motivation.error(state0).cpu()
            cnd_state = agent.network.cnd_model.preprocess(state0)
            analytic.update(ext_reward=ext_reward,
                            int_reward=int_reward,
                            ext_value=value[:, 0].unsqueeze(-1).cpu(),
                            int_value=value[:, 1].unsqueeze(-1).cpu(),
                            error=error,
                            state_space=cnd_state.norm(p=2, dim=[1, 2, 3]).unsqueeze(-1).cpu(),
                            feature_space=features.norm(p=2, dim=1, keepdim=True).cpu())

            env_indices = numpy.nonzero(numpy.squeeze(done, axis=1))[0]
            stats = analytic.reset(env_indices)
            step_counter.update(n_env)

            for i, index in enumerate(env_indices):
                # step_counter.update(int(stats['ext_reward'].step[i].item()))
                reward_avg.update(stats['ext_reward'].sum[i].item())

                print('Run {0:d} step {1:d}/{2:d} training [ext. reward {3:f} int. reward (max={4:f} mean={5:f} std={6:f}) steps {7:d}  mean reward {8:f} score {9:f} feature space (max={10:f} mean={11:f} std={12:f})]'.format(
                    trial, step_counter.steps, step_counter.limit, stats['ext_reward'].sum[i].item(), stats['int_reward'].max[i].item(), stats['int_reward'].mean[i].item(), stats['int_reward'].std[i].item(),
                    int(stats['ext_reward'].step[i].item()), reward_avg.value().item(), stats['score'].sum[i].item(), stats['feature_space'].max[i].item(), stats['feature_space'].mean[i].item(), stats['feature_space'].std[i].item()))
                print(time_estimator)
                next_state[i] = self._env.reset(index)

            state1 = self.process_state(next_state)

            reward = torch.cat([ext_reward, int_reward], dim=1)
            done = torch.tensor(1 - done, dtype=torch.float32)
            analytic.end_step()

            agent.train(state0, value, action0, probs0, state1, reward, done)

            state0 = state1
            time_estimator.update(n_env)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(config.name, config.model, trial))

        print('Saving data...')
        analytic.reset(numpy.array(range(n_env)))
        save_data = analytic.finalize()
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)
        analytic.clear()

    def run_dop_model(self, agent, trial):
        config = self._config
        n_env = config.n_env
        trial = trial + config.shift
        step_counter = StepCounter(int(config.steps * 1e6))
        time_estimator = PPOTimeEstimator(step_counter.limit)

        steps_per_episode = []
        train_ext_rewards = []
        train_ext_reward = numpy.zeros((n_env, 1), dtype=numpy.float32)
        train_int_rewards = []
        train_int_reward = numpy.zeros((n_env, 1), dtype=numpy.float32)
        train_scores = []
        train_score = numpy.zeros((n_env, 1), dtype=numpy.float32)
        train_errors = []
        train_error = numpy.zeros((n_env, 1), dtype=numpy.float32)
        train_steps = numpy.zeros((n_env, 1), dtype=numpy.int32)
        head_index_density = numpy.zeros((n_env, config.dop_heads))
        train_head_index = []
        reward_avg = RunningAverageWindow(100)
        # time_avg = RunningAverageWindow(100)

        s = numpy.zeros((n_env,) + self._env.observation_space.shape, dtype=numpy.float32)
        for i in range(n_env):
            s[i] = self._env.reset(i)

        state0 = self.process_state(s)
        ext_reward = torch.zeros((n_env, config.dop_heads, 1), dtype=torch.float32, device=config.device)

        while step_counter.running():
            with torch.no_grad():
                features0_0, features0_1 = agent.get_features(state0)
                features0_0, value, action0, probs0, features0_1, head_value, head_action, head_probs, selected_action = agent.get_action(features0_0, features0_1)
                # print(head_value[0], head_action[0], head_probs[0])
            agent.motivation.update_state_average(state0, selected_action)
            next_state, reward0_0, done0_0, info = self._env.step(agent.convert_action(selected_action.cpu()))

            reward0_1, done0_1 = agent.controller.network.aggregate_values(reward0_0, done0_0)
            head_index_density += head_action.cpu().numpy()
            error = agent.motivation.error(agent.extend_state(state0), action0)

            ext_reward.zero_()
            ext_reward = ext_reward.scatter(1, head_action.argmax(dim=1, keepdim=True).unsqueeze(-1), torch.tensor(reward0_0, dtype=torch.float32, device=config.device).unsqueeze(-1))
            int_reward = agent.motivation.reward(error).clip(0.0, 1.0).view(-1, config.dop_heads, 1)

            if info is not None and 'raw_score' in info:
                score = numpy.expand_dims(info['raw_score'], axis=1)
                train_score += score

            sel_error = agent.motivation.error(state0, selected_action)
            sel_int_reward = agent.motivation.reward(sel_error).cpu().clip(0.0, 1.0)
            sel_error = sel_error.cpu()

            train_steps += 1
            train_ext_reward += reward0_0
            train_int_reward += sel_int_reward.numpy()
            train_error += sel_error.numpy()

            env_indices = numpy.nonzero(numpy.squeeze(done0_0, axis=1))[0]

            for i in env_indices:
                if step_counter.steps + train_steps[i] > step_counter.limit:
                    train_steps[i] = step_counter.limit - step_counter.steps
                step_counter.update(train_steps[i].item())

                steps_per_episode.append(train_steps[i].item())
                train_ext_rewards.append(train_ext_reward[i].item())
                train_int_rewards.append(train_int_reward[i].item())
                train_scores.append(train_score[i].item())
                train_errors.append(train_error[i].item())
                train_head_index.append(head_index_density[i] / train_steps[i].item())
                reward_avg.update(train_ext_reward[i].item())

                if train_steps[i].item() > 0:
                    print('Run {0:d} step {1:d} training [ext. reward {2:f} int. reward {3:f} steps {4:d} ({5:f})  mean reward {6:f} density {7:s}]'.format(
                        trial, step_counter.steps, train_ext_reward[i].item(), train_int_reward[i].item(), train_steps[i].item(), train_int_reward[i].item() / train_steps[i].item(),
                        reward_avg.value().item(), numpy.array2string(head_index_density[i] / train_steps[i].item(), precision=2), train_score[i].item()))
                print(time_estimator)

                train_ext_reward[i] = 0
                train_int_reward[i] = 0
                train_score[i] = 0
                train_steps[i] = 0
                train_error[i] = 0
                head_index_density[i].fill(0)

                next_state[i] = self._env.reset(i)

            state1 = self.process_state(next_state)

            reward0_0 = torch.cat([ext_reward, int_reward], dim=2)
            reward0_1 = torch.tensor(reward0_1, dtype=torch.float32)
            done0_0 = torch.tensor(1 - done0_0, dtype=torch.float32)
            done0_1 = torch.tensor(1 - done0_1, dtype=torch.float32)

            agent.train(features0_0, features0_1, state0, value, action0, probs0, head_value, head_action, head_probs, state1, reward0_0, reward0_1, done0_0, done0_1)

            state0 = state1
            agent.controller.network.aggregator.reset(env_indices.tolist())
            time_estimator.update(n_env)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(config.name, config.model, trial))

        print('Saving data...')
        save_data = {
            'steps': numpy.array(steps_per_episode),
            'score': numpy.array(train_scores),
            're': numpy.array(train_ext_rewards),
            'ri': numpy.array(train_int_rewards),
            'error': numpy.array(train_errors),
            'hid': numpy.stack(train_head_index)
        }
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)


    def run_fwd_model(self, agent, trial):
        config = self._config
        n_env = config.n_env
        trial = trial + config.shift
        step_counter = StepCounter(int(config.steps * 1e6))

        analytic = RNDAnalytic()
        analytic.init(n_env, ext_reward=(1,), score=(1,), int_reward=(1,), error=(1,), ext_value=(1,), int_value=(1,))

        reward_avg = RunningAverageWindow(100)
        time_estimator = PPOTimeEstimator(step_counter.limit)

        s = numpy.zeros((n_env,) + self._env.observation_space.shape, dtype=numpy.float32)
        for i in range(n_env):
            s[i] = self._env.reset(i)

        state0 = self.process_state(s)

        while step_counter.running():
            with torch.no_grad():
                value, action0, probs0 = agent.get_action(state0)
            next_state, reward, done, info = self._env.step(agent.convert_action(action0.cpu()))

            ext_reward = torch.tensor(reward, dtype=torch.float32)
            int_reward = agent.motivation.reward(state0, action0, self.process_state(next_state)).cpu().clip(0.0, 1.0)

            if info is not None and 'raw_score' in info:
                score = torch.tensor(info['raw_score']).unsqueeze(-1)
                analytic.update(score=score)


            env_indices = numpy.nonzero(numpy.squeeze(done, axis=1))[0]
            stats = analytic.reset(env_indices)
            step_counter.update(n_env)

            for i, index in enumerate(env_indices):
                reward_avg.update(stats['ext_reward'].sum[i].item())

                print('Run {0:d} step {1:d}/{2:d} training [ext. reward {3:f} int. reward (max={4:f} mean={5:f} std={6:f}) steps {7:d}  mean reward {8:f} score {9:f})]'.format(
                    trial, step_counter.steps, step_counter.limit, stats['ext_reward'].sum[i].item(), stats['int_reward'].max[i].item(), stats['int_reward'].mean[i].item(), stats['int_reward'].std[i].item(),
                    int(stats['ext_reward'].step[i].item()), reward_avg.value().item(), stats['score'].sum[i].item()))
                print(time_estimator)
                next_state[i] = self._env.reset(index)

            state1 = self.process_state(next_state)

            error = agent.motivation.error(state0, action0, state1).cpu()
            analytic.update(ext_reward=ext_reward,
                            int_reward=int_reward,
                            ext_value=value[:, 0].unsqueeze(-1).cpu(),
                            int_value=value[:, 1].unsqueeze(-1).cpu(),
                            error=error)

            reward = torch.cat([ext_reward, int_reward], dim=1)
            done = torch.tensor(1 - done, dtype=torch.float32)
            analytic.end_step()

            agent.train(state0, value, action0, probs0, state1, reward, done)

            state0 = state1
            time_estimator.update(n_env)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(config.name, config.model, trial))

        print('Saving data...')
        analytic.reset(numpy.array(range(n_env)))
        save_data = analytic.finalize()
        numpy.save('ppo_{0}_{1}_{2:d}'.format(config.name, config.model, trial), save_data)
        analytic.clear()
