import torch


class VAE_ForwardModelMotivation:
    def __init__(self, network, lr, eta=1, memory_buffer=None, sample_size=0):
        self.vae = network.vae
        self._network = network
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        self._memory = memory_buffer
        self._sample_size = sample_size
        self._eta = eta

    def train(self, state0, action, state1):
        if self._memory is not None:
            if len(self._memory) > self._sample_size:
                sample = self._memory.sample(self._sample_size)

                states = torch.stack(sample.state)
                next_states = torch.stack(sample.next_state)
                actions = torch.stack(sample.action)

                self._optimizer.zero_grad()
                loss = self.error(states, actions, next_states).mean() + self._network.vae.loss_function(states) * 10
                loss.backward()
                self._optimizer.step()

    def error(self, state0, action, state1):
        prediction = self._network(state0, action)
        mu, logvar = self._network.vae.encode(state1)
        target = self._network.vae.reparameterize(mu, logvar)
        dim = len(prediction.shape) - 1
        error = torch.mean(torch.pow(prediction - target, 2), dim=dim).unsqueeze(dim)

        return error

    def reward(self, state0=None, action=None, state1=None, error=None):
        reward = 0
        if error is None:
            reward = torch.tanh(self.error(state0, action, state1).detach())
        else:
            reward = torch.tanh(error)

        return reward * self._eta

    def save(self, path):
        torch.save(self._network.state_dict(), path + '_fm.pth')

    def load(self, path):
        self._network.load_state_dict(torch.load(path + '_fm.pth'))
