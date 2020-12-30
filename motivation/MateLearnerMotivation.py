import abc

import torch


class MetaLearnerModel(torch.nn.Module):
    @abc.abstractmethod
    def __init__(self, state_dim, action_dim, config):
        super(MetaLearnerModel, self).__init__()

    @abc.abstractmethod
    def forward(self, state, action):
        raise NotImplementedError


class MetaLearnerMotivation:
    def __init__(self, network, forward_model, lr, state_dim, variant='A', window=1, eta=1.0, memory_buffer=None, sample_size=0):
        self._forward_model = forward_model
        self._network = network
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        self._memory = memory_buffer
        self._sample_size = sample_size
        self._variant = variant
        self._eta = eta
        # self._sigma = state_dim * 0.005
        self._sigma = 1e-2

        if self._variant == 'E':
            self._window = window
            self._window_index = 0
            self._error_buffer = torch.zeros(window, dtype=torch.float32)

    def train(self, state0, action, state1):
        self._forward_model.train(state0, action, state1)

        if self._variant == 'E':
            self._error_buffer[self._window_index] = self.error(state0, action)
            self._window_index += 1
            if self._window_index == self._window:
                self._window_index = 0

        if self._memory is not None:
            if len(self._memory) > self._sample_size:
                sample = self._memory.sample(self._sample_size)

                states = torch.stack(sample.state)
                next_states = torch.stack(sample.next_state)
                actions = torch.stack(sample.action)

                self._optimizer.zero_grad()
                error = self._forward_model.error(states, actions, next_states)
                loss = torch.nn.functional.mse_loss(self._network(states, actions), error)
                loss.backward()
                self._optimizer.step()
        else:
            self._optimizer.zero_grad()
            error = self._forward_model.error(state0, action, state1)
            loss = torch.nn.functional.mse_loss(self._network(state0, action), error)
            loss.backward()
            self._optimizer.step()

    def error(self, state0, action):
        with torch.no_grad():
            error = self._network(state0, action)
        return error

    def mean_error(self):
        return self._error_buffer.mean()

    def reward(self, state0, action, state1):
        k = 1

        if self._variant == 'E':
            mean_error = self._forward_model.mean_error() + 1e-8
            mean_error_estimate = self.mean_error() + 1e-8

            reward = torch.max(torch.tanh(mean_error / mean_error_estimate + mean_error_estimate / mean_error - 2))
            reward = torch.max(reward, self._forward_model.reward())
        else:
            error = self._forward_model.error(state0, action, state1)
            error_estimate = self.error(state0, action)

            reward = None
            if self._variant == 'A':
                mask = torch.gt(torch.abs(error - error_estimate), torch.ones_like(error) * self._sigma).type(torch.float32)
                reward = torch.max(torch.tanh(error / error_estimate + error_estimate / error - 2) * mask, torch.zeros_like(error))
                reward = torch.max(reward, self._forward_model.reward(error=error))

            if self._variant == 'B':
                reward = torch.exp(k * torch.abs(error - error_estimate)) - 1
                reward = torch.max(reward, self._forward_model.reward(error=error))

            if self._variant == 'C':
                mask = torch.gt(torch.abs(error - error_estimate), torch.ones_like(error) * self._sigma).type(torch.float32)
                reward = torch.max(torch.tanh(error / error_estimate + error_estimate / error - 2) * mask, torch.zeros_like(error))

            if self._variant == 'D':
                mask = torch.gt(torch.abs(error - error_estimate), torch.ones_like(error) * self._sigma).type(torch.float32)
                reward =torch.max(torch.tanh(error / error_estimate + error_estimate / error - 2) * mask, torch.zeros_like(error))
                reward = torch.max(reward, self._forward_model.reward(error=error) - error_estimate)

        return reward * self._eta

    def raw_data(self, state0, action, state1):
        k = 1

        if self._variant == 'E':
            pe_reward = self._forward_model.reward()
            error = self._forward_model.mean_error() + 1e-8
            error_estimate = self.mean_error() + 1e-8

            ps_reward = torch.max(torch.tanh(error / error_estimate + error_estimate / error - 2))
            reward = torch.max(ps_reward, pe_reward)
        else:
            error = self._forward_model.error(state0, action, state1)
            pe_reward = self._forward_model.reward(error=error)
            error_estimate = self.error(state0, action)

            reward = None
            if self._variant == 'A':
                mask = torch.gt(torch.abs(error - error_estimate), torch.ones_like(error) * self._sigma).type(torch.float32)
                ps_reward = torch.max(torch.tanh(error / error_estimate + error_estimate / error - 2) * mask, torch.zeros_like(error))
                reward = torch.max(ps_reward, pe_reward)

            if self._variant == 'B':
                ps_reward = torch.exp(k * torch.abs(error - error_estimate)) - 1
                reward = torch.max(ps_reward, pe_reward)

            if self._variant == 'C':
                mask = torch.gt(torch.abs(error - error_estimate), torch.ones_like(error) * self._sigma).type(torch.float32)
                reward = torch.max(torch.tanh(error / error_estimate + error_estimate / error - 2) * mask, torch.zeros_like(error))
                ps_reward = reward

            if self._variant == 'D':
                mask = torch.gt(torch.abs(error - error_estimate), torch.ones_like(error) * self._sigma).type(torch.float32)
                ps_reward = torch.max(torch.tanh(error / error_estimate + error_estimate / error - 2) * mask, torch.zeros_like(error))
                reward = torch.max(ps_reward, pe_reward - error_estimate)

        return error, error_estimate, pe_reward, ps_reward, reward * self._eta

    def save(self, path):
        self._forward_model.save(path)
        torch.save(self._network.state_dict(), path + '_mc.pth')

    def load(self, path):
        self._forward_model.load(path)
        self._network.load_state_dict(torch.load(path + '_mc.pth'))

    def get_forward_model(self):
        return self._forward_model
