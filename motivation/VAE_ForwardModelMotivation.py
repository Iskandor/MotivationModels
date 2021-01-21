import torch


class VAE_ForwardModelMotivation:
    def __init__(self, fm_network, fm_lr, vae_network, vae_lr, vae_sample_size, eta=1, memory_buffer=None, sample_size=0):
        self._fm_network = fm_network
        self._fm_optimizer = torch.optim.Adam(self._fm_network.parameters(), lr=fm_lr)
        self._vae_network = vae_network
        self._vae_optimizer = torch.optim.Adam(self._vae_network.parameters(), lr=vae_lr)
        self._vae_sample_size = vae_sample_size
        self._vae_sample_counter = 0

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
            loss = self.error(states, actions, next_states).mean()
            loss.backward()
            self._fm_optimizer.step()

        self._vae_sample_counter += 1
        if self._vae_sample_counter > self._vae_sample_size and len(self._memory) > self._vae_sample_size:
            print('Training VAE')
            sample = self._memory.sample(self._vae_sample_size)
            states = torch.stack(sample.state)

            self._vae_optimizer.zero_grad()
            loss = self._vae_network.loss_function(states)
            loss.backward()
            self._vae_optimizer.step()
            self._vae_sample_counter = 0

    def error(self, state0, action, state1):
        mu, logvar = self._vae_network.encode(state0)
        vae_state0 = self._vae_network.reparameterize(mu, logvar).detach()

        prediction = self._fm_network(vae_state0, action)
        mu, logvar = self._vae_network.encode(state1)
        vae_state1 = self._vae_network.reparameterize(mu, logvar).detach()
        dim = len(prediction.shape) - 1
        error = torch.mean(torch.pow(prediction - vae_state1, 2), dim=dim).unsqueeze(dim)

        return error

    def reward(self, state0=None, action=None, state1=None, error=None):
        reward = 0
        if error is None:
            reward = torch.tanh(self.error(state0, action, state1).detach())
        else:
            reward = torch.tanh(error)

        return reward * self._eta

    def get_vae(self):
        return self._vae_network

    def save(self, path):
        torch.save(self._fm_network.state_dict(), path + '_fm.pth')

    def load(self, path):
        self._fm_network.load_state_dict(torch.load(path + '_fm.pth'))
