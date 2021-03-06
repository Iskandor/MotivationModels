import torch


class VAE_ForwardModelMotivation:
    def __init__(self, fm_network, fm_lr, eta=1, memory_buffer=None, sample_size=0):
        self._fm_network = fm_network
        self._fm_optimizer = torch.optim.Adam(self._fm_network.parameters(), lr=fm_lr)

        self._memory = memory_buffer
        self._sample_size = sample_size
        self._eta = eta

    def train(self, state0, action, state1):
        if len(self._memory) > self._sample_size:
            sample = self._memory.sample(self._sample_size)

            states = torch.stack(sample.state)
            next_states = torch.stack(sample.next_state)
            actions = torch.stack(sample.action)

            self._fm_optimizer.zero_grad()
            loss = self._fm_network.loss_function(states, actions, next_states)
            loss.backward()
            self._fm_optimizer.step()

    def error(self, state0, action, state1):
        prediction, _, _ = self._fm_network(state0, action)
        dim = len(prediction.shape) - 1
        error = torch.mean(torch.pow(prediction - state1, 2), dim=dim).unsqueeze(dim)

        return error

    def reward(self, state0=None, action=None, state1=None, error=None):
        reward = 0
        if error is None:
            reward = torch.tanh(self.error(state0, action, state1).detach())
        else:
            reward = torch.tanh(error)

        return reward * self._eta

    def get_fm_network(self):
        return self._fm_network

    def save(self, path):
        torch.save(self._fm_network.state_dict(), path + '_fm.pth')

    def load(self, path):
        self._fm_network.load_state_dict(torch.load(path + '_fm.pth'))
