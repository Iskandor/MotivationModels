import torch


class Encoder:
    def __init__(self, network, lr, device='cpu'):
        self.network = network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.device = device

    def train(self, memory, indices):
        if indices:
            sample, size = memory.sample_batches(indices)

            for i in range(size):
                states = sample.state[i].to(self.device)
                next_states = sample.next_state[i].to(self.device)

                self.optimizer.zero_grad()
                loss = self.network.loss_function(states, next_states)
                loss.backward()
                self.optimizer.step()
