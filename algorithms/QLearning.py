import abc

import torch


class QCritic(torch.nn.Module):
    @abc.abstractmethod
    def __init__(self, state_dim, action_dim):
        super(QCritic, self).__init__()

    @abc.abstractmethod
    def forward(self, state):
        raise NotImplementedError


class QLearning:
    def __init__(self, critic_class, state_dim, action_dim, critic_lr, gamma, weight_decay=0):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._critic = critic_class(state_dim, action_dim)
        self._gamma = gamma

        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=critic_lr, weight_decay=weight_decay)

    def get_action(self, state):
        action = torch.zeros(self._action_dim, dtype=torch.float32).scatter(dim=0, index=torch.tensor([self._critic(state).argmax()]), src=torch.tensor([1], dtype=torch.float32))
        return action

    def train(self, state0, action0, state1, reward, done):
        current_q_values = self._critic(state0).gather(0, action0)
        max_next_q_values = self._critic(state1).max(0).values
        expected_q_values = reward + (1 - done) * (self._gamma * max_next_q_values)

        self._critic_optimizer.zero_grad()
        value_loss = torch.nn.functional.mse_loss(current_q_values, expected_q_values)
        value_loss.backward()
        self._critic_optimizer.step()

    def activate(self, state):
        return self._critic(state)
