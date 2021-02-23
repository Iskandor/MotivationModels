import abc

import torch


class MetaCriticMotivation:
    def __init__(self, metacritic, lr, variant='A', eta=1.0, memory_buffer=None, sample_size=0):
        self._metacritic = metacritic
        self._optimizer = torch.optim.Adam(self._metacritic.parameters(), lr=lr)
        self._memory = memory_buffer
        self._sample_size = sample_size
        self._variant = variant
        self._eta = eta
        self._sigma = 1e-2

    def train(self, state0, action, state1):
        if self._memory is not None:
            if len(self._memory) > self._sample_size:
                sample = self._memory.sample(self._sample_size)

                states = torch.stack(sample.state).squeeze(1)
                next_states = torch.stack(sample.next_state).squeeze(1)
                actions = torch.stack(sample.action).squeeze(1)

                self._optimizer.zero_grad()
                loss = self._metacritic.loss_function(states, actions, next_states)
                loss.backward()
                self._optimizer.step()
        else:
            self._optimizer.zero_grad()
            loss = self._metacritic.loss_function(state0, action, state1)
            loss.backward()
            self._optimizer.step()

    def error_estimate(self, state0, action):
        with torch.no_grad():
            error = self._metacritic.error_estimate(state0, action)
        return error

    def error(self, state0, action, state1):
        with torch.no_grad():
            error = self._metacritic.error(state0, action, state1)
        return error

    def reward(self, state0, action, state1):
        k = 1

        error = self._metacritic.error(state0, action, state1)
        error_estimate = self._metacritic.error_estimate(state0, action)

        reward = None
        if self._variant == 'A':
            mask = torch.gt(torch.abs(error - error_estimate), torch.ones_like(error) * self._sigma).type(torch.float32)
            reward = torch.max(torch.tanh(error / error_estimate + error_estimate / error - 2) * mask, torch.zeros_like(error))
            reward = torch.max(reward, torch.tanh(error))

        if self._variant == 'B':
            reward = torch.exp(k * torch.abs(error - error_estimate)) - 1
            reward = torch.max(reward, torch.tanh(error))

        if self._variant == 'C':
            mask = torch.gt(torch.abs(error - error_estimate), torch.ones_like(error) * self._sigma).type(torch.float32)
            reward = torch.max(torch.tanh(error / error_estimate + error_estimate / error - 2) * mask, torch.zeros_like(error))

        if self._variant == 'D':
            mask = torch.gt(torch.abs(error - error_estimate), torch.ones_like(error) * self._sigma).type(torch.float32)
            reward =torch.max(torch.tanh(error / error_estimate + error_estimate / error - 2) * mask, torch.zeros_like(error))
            reward = torch.max(reward, torch.tanh(error) - error_estimate)

        return reward * self._eta

    def raw_data(self, state0, action, state1):
        k = 1

        error = self.error(state0, action, state1)
        pe_reward = torch.tanh(error)
        error_estimate = self.error_estimate(state0, action)
        ps_reward = 0

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
        torch.save(self._metacritic.state_dict(), path + '_mc.pth')

    def load(self, path):
        self._metacritic.load_state_dict(torch.load(path + '_mc.pth'))
