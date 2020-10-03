import torch


class TD:
    def __init__(self, network, critic_lr, gamma, weight_decay=0):
        self._critic = network
        self._gamma = gamma

        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=critic_lr, weight_decay=weight_decay)

    def train(self, state0, state1, reward, done):
        current_v_values = self._critic(state0)
        next_v_values = self._critic(state1).detach()
        expected_v_values = reward + (1 - done) * (self._gamma * next_v_values)

        self._critic_optimizer.zero_grad()
        value_loss = torch.nn.functional.mse_loss(current_v_values, expected_v_values)
        value_loss.backward()
        self._critic_optimizer.step()

    def td_error(self, state0, state1, reward, done):
        with torch.no_grad():
            current_v_values = self._critic(state0)
            next_v_values = self._critic(state1)
            expected_v_values = reward + (1 - done) * (self._gamma * next_v_values) - current_v_values

        return expected_v_values

    def activate(self, state):
        return self._critic(state)
