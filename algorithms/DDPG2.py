import copy

import torch


class DDPG2:
    def __init__(self, agent, actor_lr, critic_lr, gamma, tau, memory_buffer, sample_size):
        self._agent = agent
        self._motivation_module = None
        self._memory = memory_buffer
        self._sample_size = sample_size
        self._gamma = gamma
        self._tau = tau

        self._critic_optimizer = torch.optim.Adam(self._agent.critic.parameters(), lr=critic_lr)
        self._actor_optimizer = torch.optim.Adam(self._agent.actor.parameters(), lr=actor_lr)

    def add_motivation_module(self, motivation_module):
        self._motivation_module = motivation_module

    def get_motivation_module(self):
        return self._motivation_module

    def get_action(self, state):
        with torch.no_grad():
            action = self._agent.action(state)
        return action

    def get_value(self, state, action):
        with torch.no_grad():
            value = self._agent.critic(state, action)
        return value

    def train(self, state0, action0, state1, reward, done):
        self._memory.add(state0, action0, state1, reward, done)

        if len(self._memory) > self._sample_size:
            sample = self._memory.sample(self._sample_size)

            states = torch.stack(sample.state).squeeze(1)
            next_states = torch.stack(sample.next_state).squeeze(1)
            actions = torch.stack(sample.action).squeeze(1)
            rewards = torch.stack(sample.reward)
            masks = torch.stack(sample.mask)

            if self._motivation_module:
                rewards += self._motivation_module.reward(states, actions, next_states)

            expected_values = rewards + masks * self._gamma * self._agent.value_target(next_states, self._agent.action_target(next_states).detach()).detach()

            self._critic_optimizer.zero_grad()
            loss = self._agent.critic_loss(states, actions, next_states, expected_values)
            loss.backward()
            self._critic_optimizer.step()

            self._actor_optimizer.zero_grad()
            loss = self._agent.actor_loss(states)
            loss.backward()
            self._actor_optimizer.step()

            self._agent.soft_update(self._tau)

    def save(self, path):
        torch.save(self._agent.state_dict(), path + '.pth')
        if self._motivation_module:
            self._motivation_module.save(path)

    def load(self, path):
        self._agent.load_state_dict(torch.load(path + '.pth'))
        if self._motivation_module:
            self._motivation_module.load(path)
