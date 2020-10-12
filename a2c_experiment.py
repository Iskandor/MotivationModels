import numpy
import torch
from etaprogress.progress import ProgressBar


class ExperimentA2C:
    def __init__(self, env_name, env, config):
        self._env_name = env_name
        self._env = env
        self._config = config
        self._agent = None

    def run_baseline(self, agent, trial):
        config = self._config
        bar = ProgressBar(config.episodes, max_width=40)
        ext_rewards = numpy.zeros(config.episodes)

        for e in range(config.episodes):
            bar.numerator = e
            # state0 = torch.tensor(self._env.reset(), dtype=torch.float32).unsqueeze(0).to(config.device)
            # done = False
            #
            # if e % 100 == 0:
            #     while not done:
            #         self._env.render()
            #         action0, _ = agent.get_action(state0)
            #         next_state, reward, done, info = self._env.step(action0)
            #         state0 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(config.device)

            state0 = torch.tensor(self._env.reset(), dtype=torch.float32).unsqueeze(0).to(config.device)
            done = False
            total_reward = 0
            train_steps = 0

            while not done:
                action0, log_prob = agent.get_action(state0)
                next_state, reward, done, info = self._env.step(action0)
                total_reward += reward
                train_steps += 1
                agent.train(state0, log_prob, reward, done)
                state0 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(config.device)

            ext_rewards[e] = total_reward
            print('Episode {0:d} training [ext. reward {1:f} steps {2:d}]'.format(e, total_reward, train_steps))
            print(bar)

        agent.save('./models/{0:s}_{1}_{2:d}'.format(self._env_name, config.model, trial))
        numpy.save('a2c_{0}_{1}_{2:d}_re'.format(config.name, config.model, trial), ext_rewards)
