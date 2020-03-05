import abc
import copy
import torch
from ReplayBuffer import ReplayBuffer


class DQNCritic(torch.nn.Module):
    @abc.abstractmethod
    def __init__(self, state_dim, action_dim):
        super(DQNCritic, self).__init__()

    @abc.abstractmethod
    def forward(self, state):
        raise NotImplementedError


class DQN:
    def __init__(self, critic_class, state_dim, action_dim, memory_size, sample_size, critic_lr, gamma, weight_decay=0):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._critic = critic_class(state_dim, action_dim)
        self._critic_target = critic_class(state_dim, action_dim)
        self._memory = ReplayBuffer(memory_size)
        self._sample_size = sample_size
        self._gamma = gamma
        self._update_step = 0

        self._hard_update(self._critic_target, self._critic)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=critic_lr, weight_decay=weight_decay)

    def get_action(self, state):
        o = self.activate(state)
        return self.activate(state).argmax(0).item()

    def train(self, state0, action0, state1, reward, done):
        self._memory.add(state0, action0, state1, reward, done)

        if len(self._memory) > self._sample_size:
            sample = self._memory.sample(self._sample_size)

            states = torch.stack(sample.state)
            next_states = torch.stack(sample.next_state)
            actions = torch.Tensor(sample.action).long().unsqueeze(1)
            rewards = torch.Tensor(sample.reward)
            masks = torch.Tensor(sample.mask)

            Qs0a = self._critic(states).gather(1, actions).squeeze()
            Qs1max = self._critic_target(next_states).max(1)[0]
            target = rewards + masks * self._gamma * Qs1max

            loss = torch.nn.functional.mse_loss(Qs0a, target.detach())
            self._critic_optimizer.zero_grad()
            loss.backward()
            self._critic_optimizer.step()

            self._update_step += 1
            if self._update_step == 100:
                self._update_step = 0
                self._hard_update(self._critic_target, self._critic)

    def activate(self, state):
        return self._critic(state)

    def _hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
