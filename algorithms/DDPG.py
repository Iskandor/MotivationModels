import abc
import copy

import torch


class DDPGCritic(torch.nn.Module):
    @abc.abstractmethod
    def __init__(self, state_dim, action_dim):
        super(DDPGCritic, self).__init__()

    @abc.abstractmethod
    def forward(self, state, action):
        raise NotImplementedError


class DDPGActor(torch.nn.Module):
    @abc.abstractmethod
    def __init__(self, state_dim, action_dim):
        super(DDPGActor, self).__init__()

    @abc.abstractmethod
    def forward(self, state):
        raise NotImplementedError


class DDPG:
    def __init__(self, actor, critic, actor_lr, critic_lr, gamma, tau, memory_buffer, sample_size):
        self._actor = actor
        self._critic = critic
        self._actor_target = copy.deepcopy(actor)
        self._critic_target = copy.deepcopy(critic)
        self._motivation_module = None
        self._memory = memory_buffer
        self._sample_size = sample_size
        self._gamma = gamma
        self._tau = tau

        self._hard_update(self._actor_target, self._actor)
        self._hard_update(self._critic_target, self._critic)

        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=actor_lr)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=critic_lr)

        self._gpu_enabled = False

        self.int_reward_filtered = 0.0

    def add_motivation_module(self, motivation_module):
        self._motivation_module = motivation_module

    def get_motivation_module(self):
        return self._motivation_module

    def get_action(self, state):
        with torch.no_grad():
            action = self._actor(state).detach()
        return action

    def get_value(self, state, action):
        with torch.no_grad():
            value = self._critic(state, action).detach()
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

            if self._gpu_enabled:
                states = states.cuda()
                next_states = next_states.cuda()
                actions = actions.cuda()
                rewards = rewards.cuda()
                masks = masks.cuda()

            if self._critic.heads == 1:
                if self._motivation_module:
                    rewards += self._motivation_module.reward(states, actions, next_states)

                expected_values = rewards + masks * self._gamma * self._critic_target(next_states, self._actor_target(next_states).detach()).detach()

                self._critic_optimizer.zero_grad()
                value_loss = torch.nn.functional.mse_loss(self._critic(states, actions), expected_values)
                value_loss.backward()
                self._critic_optimizer.step()

                self._actor_optimizer.zero_grad()
                policy_loss = -self._critic(states, self._actor(states))
                policy_loss = policy_loss.mean()
                policy_loss.backward()
                self._actor_optimizer.step()

            if self._critic.heads == 2:
                rewards_internal = self._motivation_module.reward(states, actions, next_states)
                value_re, value_ri = self._critic(states, actions)
                value_re_target, value_ri_target = self._critic_target(next_states, self._actor_target(next_states).detach())

                expected_values_re = rewards + masks * self._gamma * value_re_target.detach()
                expected_values_ri = rewards_internal + masks * self._gamma * value_ri_target.detach()

                self._critic_optimizer.zero_grad()
                value_loss = torch.nn.functional.mse_loss(value_re, expected_values_re) + torch.nn.functional.mse_loss(value_ri, expected_values_ri)
                value_loss.backward()
                self._critic_optimizer.step()

                self._actor_optimizer.zero_grad()
                policy_target_re, policy_target_ri = self._critic(states, self._actor(states))
                policy_loss = policy_target_re + policy_target_ri
                policy_loss = -policy_loss.mean()
                policy_loss.backward()
                self._actor_optimizer.step()

            self._soft_update(self._actor_target, self._actor, self._tau)
            self._soft_update(self._critic_target, self._critic, self._tau)

    @staticmethod
    def _soft_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    @staticmethod
    def _hard_update(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def enable_gpu(self):
        self._actor.cuda()
        self._critic.cuda()
        self._actor_target.cuda()
        self._critic_target.cuda()
        self._gpu_enabled = True

    def disable_gpu(self):
        self._actor.cpu()
        self._critic.cpu()
        self._actor_target.cpu()
        self._critic_target.cpu()
        self._gpu_enabled = False

    def save(self, path):
        torch.save(self._actor.state_dict(), path + '_actor.pth')
        torch.save(self._critic.state_dict(), path + '_critic.pth')
        if self._motivation_module:
            self._motivation_module.save(path)

    def load(self, path):
        self._actor.load_state_dict(torch.load(path + '_actor.pth'))
        self._critic.load_state_dict(torch.load(path + '_critic.pth'))
        if self._motivation_module:
            self._motivation_module.load(path)
