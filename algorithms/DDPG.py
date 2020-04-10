import abc

import torch

from algorithms.ReplayBuffer import ReplayBuffer


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
    def __init__(self, actor, critic, state_dim, action_dim, memory_size, sample_size, actor_lr, critic_lr, gamma, tau, weight_decay=0, motivation_module=None):
        self._actor = actor(state_dim, action_dim)
        self._critic = critic(state_dim, action_dim)
        self._actor_target = actor(state_dim, action_dim)
        self._critic_target = critic(state_dim, action_dim)
        self._motivation_module = motivation_module
        self._memory = ReplayBuffer(memory_size)
        self._sample_size = sample_size
        self._gamma = gamma
        self._tau = tau

        self._hard_update(self._actor_target, self._actor)
        self._hard_update(self._critic_target, self._critic)

        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=actor_lr)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=critic_lr, weight_decay=weight_decay)

        self._gpu_enabled = False

    def get_action(self, state):
        return self._actor(state)

    def train(self, state0, action0, state1, reward, done):
        self._memory.add(state0, action0, state1, reward, done)

        if len(self._memory) > self._sample_size:
            sample = self._memory.sample(self._sample_size)

            states = torch.stack(sample.state)
            next_states = torch.stack(sample.next_state)
            actions = torch.stack(sample.action).detach()
            rewards = torch.Tensor(sample.reward).unsqueeze(1)
            masks = torch.Tensor(sample.mask).unsqueeze(1)

            if self._gpu_enabled:
                states = states.cuda()
                next_states = next_states.cuda()
                actions = actions.cuda()
                rewards = rewards.cuda()
                masks = masks.cuda()

            if self._motivation_module is not None:
                int_reward = self._motivation_module.reward(states, actions, next_states)
                rewards += int_reward
                #self._motivation_module.train(states, actions, next_states)

            expected_values = rewards + masks * self._gamma * self._critic_target(next_states, self._actor_target(next_states)).detach()

            self._critic_optimizer.zero_grad()
            value_loss = torch.nn.functional.mse_loss(self._critic(states, actions), expected_values)
            value_loss.backward()
            self._critic_optimizer.step()

            self._actor_optimizer.zero_grad()
            policy_loss = -self._critic(states, self._actor(states))
            policy_loss = policy_loss.mean()
            policy_loss.backward()
            self._actor_optimizer.step()

            self._soft_update(self._actor_target, self._actor, self._tau)
            self._soft_update(self._critic_target, self._critic, self._tau)

    def _soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def _hard_update(self, target, source):
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
