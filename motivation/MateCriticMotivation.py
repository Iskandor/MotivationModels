import torch


class MetaCriticMotivation:
    def __init__(self, network, lr, variant='A', eta=1.0, memory_buffer=None, sample_size=0, device='cpu'):
        self.network = network
        self._optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self._memory = memory_buffer
        self._sample_size = sample_size
        self._variant = variant
        self._eta = eta
        self._sigma = 1e-2
        self._device = device

    def train(self, indices):
        if indices:
            sample = self._memory.sample(indices)

            states = torch.stack(sample.state).squeeze(1)
            next_states = torch.stack(sample.next_state).squeeze(1)
            actions = torch.stack(sample.action).squeeze(1)

            self._optimizer.zero_grad()
            loss = self.network.metacritic_model.loss_function(states, actions, next_states)
            loss.backward()
            self._optimizer.step()

    def error(self, state0, action, state1):
        error, error_estimate = self.network.metacritic_model.error(state0, action, state1)
        return error, error_estimate

    def reward_sample(self, indices):
        sample = self._memory.sample(indices)

        states = torch.stack(sample.state).squeeze(1)
        next_states = torch.stack(sample.next_state).squeeze(1)
        actions = torch.stack(sample.action).squeeze(1)

        return self.reward(states, actions, next_states)

    def reward(self, state0, action, state1):
        k = 1

        error, error_estimate = self.network.metacritic_model.error(state0, action, state1)

        reward = None
        if self._variant == 'A':
            mask = torch.gt(torch.abs(error - error_estimate), torch.ones_like(error) * self._sigma).type(torch.float32)
            reward = torch.relu(torch.tanh(error / error_estimate + error_estimate / error - 2) * mask)
            reward = torch.max(reward, torch.tanh(error))

        if self._variant == 'B':
            reward = torch.exp(k * torch.abs(error - error_estimate)) - 1
            reward = torch.max(reward, torch.tanh(error))

        if self._variant == 'C':
            mask = torch.gt(torch.abs(error - error_estimate), torch.ones_like(error) * self._sigma).type(torch.float32)
            reward = torch.relu(torch.tanh(error / error_estimate + error_estimate / error - 2) * mask)

        if self._variant == 'D':
            mask = torch.gt(torch.abs(error - error_estimate), torch.ones_like(error) * self._sigma).type(torch.float32)
            reward = torch.relu(torch.tanh(error / error_estimate + error_estimate / error - 2) * mask)
            reward = torch.max(reward, torch.tanh(error) - error_estimate)

        return reward * self._eta

    def raw_data(self, state0, action, state1):
        k = 1

        error, error_estimate = self.error(state0, action, state1)
        pe_reward = torch.tanh(error)
        ps_reward = 0

        reward = None
        if self._variant == 'A':
            mask = torch.gt(torch.abs(error - error_estimate), torch.ones_like(error) * self._sigma).type(torch.float32)
            ps_reward = torch.relu(torch.tanh(error / error_estimate + error_estimate / error - 2) * mask)
            reward = torch.max(ps_reward, pe_reward)

        if self._variant == 'B':
            ps_reward = torch.exp(k * torch.abs(error - error_estimate)) - 1
            reward = torch.max(ps_reward, pe_reward)

        if self._variant == 'C':
            mask = torch.gt(torch.abs(error - error_estimate), torch.ones_like(error) * self._sigma).type(torch.float32)
            reward = torch.relu(torch.tanh(error / error_estimate + error_estimate / error - 2) * mask)
            ps_reward = reward

        if self._variant == 'D':
            mask = torch.gt(torch.abs(error - error_estimate), torch.ones_like(error) * self._sigma).type(torch.float32)
            ps_reward = torch.relu(torch.tanh(error / error_estimate + error_estimate / error - 2) * mask)
            reward = torch.max(ps_reward, pe_reward - error_estimate)

        return error, error_estimate, pe_reward, ps_reward, reward * self._eta
