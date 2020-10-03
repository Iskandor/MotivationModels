import abc

import torch


class TestActor(torch.nn.Module):
    @abc.abstractmethod
    def __init__(self, state_dim, action_dim):
        super(TestActor, self).__init__()

    @abc.abstractmethod
    def forward(self, state):
        raise NotImplementedError


class Actor(TestActor):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__(state_dim, action_dim)

        self._hidden0 = torch.nn.Linear(state_dim, 64)
        self._hidden1 = torch.nn.Linear(64, 32)
        self._output = torch.nn.Linear(32, action_dim)

        self.init()

    def forward(self, state):
        x = state
        x = torch.relu(self._hidden0(x))
        x = torch.relu(self._hidden1(x))
        policy = torch.tanh(self._output(x))
        return policy

    def init(self):
        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-1, 3e-1)


if __name__ == '__main__':
    state_dim = 22
    action_dim = 4
    actor = Actor(state_dim, action_dim)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)

    for i in range(6000):
        print("Episode {0}".format(i))
        for e in range(1000):
            input = torch.rand((64, state_dim))
            target = torch.rand((64, action_dim))

            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(actor(input), target)
            loss.backward()
            optimizer.step()