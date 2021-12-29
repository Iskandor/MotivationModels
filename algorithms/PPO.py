import time

import torch

from utils import *


class PPO:
    def __init__(self, network, lr, actor_loss_weight, critic_loss_weight, batch_size, trajectory_size, memory, p_beta, p_gamma,
                 ppo_epochs=10, p_epsilon=0.1, p_lambda=0.95, weight_decay=0, device='cpu', n_env=1, motivation=False):
        self._network = network
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr, weight_decay=weight_decay)
        self._beta = p_beta
        self._gamma = [float(g) for g in p_gamma.split(',')]
        self._epsilon = p_epsilon
        self._lambda = p_lambda
        self._device = device
        self._batch_size = batch_size
        self._trajectory_size = trajectory_size
        self._actor_loss_weight = actor_loss_weight
        self._critic_loss_weight = critic_loss_weight

        self._trajectory = []
        self._ppo_epochs = ppo_epochs
        self._motivation = motivation
        self._n_env = n_env

        self._memory = memory

        if self._n_env > 1:
            self._trajectories = [[] for _ in range(self._n_env)]

    def train(self, indices):
        if indices:
            start = time.time()
            sample = self._memory.sample(indices, False)

            states = sample.state
            values = sample.value
            actions = sample.action
            probs = sample.prob
            rewards = sample.reward
            dones = sample.mask

            if self._motivation:
                # mean = self._motivation.reward_stats.mean
                # std = self._motivation.reward_stats.std
                ext_reward = rewards[:, :, 0].unsqueeze(-1)
                # int_reward = (rewards[:, :, 1].unsqueeze(-1) - mean) / std
                int_reward = rewards[:, :, 1].unsqueeze(-1)
                ext_ref_values, ext_adv_values = self.calc_advantage(values[:, :, 0].unsqueeze(-1), ext_reward, dones, self._gamma[0])
                int_ref_values, int_adv_values = self.calc_advantage(values[:, :, 1].unsqueeze(-1), int_reward, dones, self._gamma[1])
                adv_values = ext_adv_values * 2.0 + int_adv_values * 1.0
                ref_values = torch.cat([ext_ref_values, int_ref_values], dim=2)
            else:
                ref_values, adv_values = self.calc_advantage(values, rewards, dones, self._gamma[0])

            permutation = torch.randperm(self._trajectory_size)

            states = states.reshape(-1, *states.shape[2:])[permutation]
            actions = actions.reshape(-1, *actions.shape[2:])[permutation]
            probs = probs.reshape(-1, *probs.shape[2:])[permutation]
            adv_values = adv_values.reshape(-1, *adv_values.shape[2:])[permutation]
            ref_values = ref_values.reshape(-1, *ref_values.shape[2:])[permutation]

            self._train(states, actions, probs, adv_values, ref_values)

            end = time.time()
            print("Trajectory {0:d} batch size {1:d} epochs {2:d} training time {3:.2f}s".format(self._trajectory_size, self._batch_size, self._ppo_epochs, end - start))

    def _train(self, states, actions, probs, adv_values, ref_values):
        # adv_values = (adv_values - torch.mean(adv_values)) / (torch.std(adv_values) + 1e-8)

        for epoch in range(self._ppo_epochs):
            for batch_ofs in range(0, self._trajectory_size, self._batch_size):
                batch_l = batch_ofs + self._batch_size
                states_v = states[batch_ofs:batch_l]
                actions_v = actions[batch_ofs:batch_l]
                probs_v = probs[batch_ofs:batch_l]
                batch_adv_v = adv_values[batch_ofs:batch_l]
                batch_ref_v = ref_values[batch_ofs:batch_l]

                self._optimizer.zero_grad()
                loss = self.calc_loss(states_v.to(self._device), batch_ref_v.to(self._device), batch_adv_v.to(self._device), actions_v.to(self._device), probs_v.to(self._device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=0.5)
                self._optimizer.step()

    def calc_loss(self, states, ref_value, adv_value, old_actions, old_probs):
        values, _, probs = self._network(states)

        if self._motivation:
            ext_value = values[:, 0]
            int_value = values[:, 1]
            ext_ref_value = ref_value[:, 0]
            int_ref_value = ref_value[:, 1]

            loss_ext_value = torch.nn.functional.mse_loss(ext_value, ext_ref_value)
            loss_int_value = torch.nn.functional.mse_loss(int_value, int_ref_value)
            loss_value = loss_ext_value + loss_int_value
        else:
            loss_value = torch.nn.functional.mse_loss(values, ref_value)

        log_probs = self._network.actor.log_prob(probs, old_actions)
        old_logprobs = self._network.actor.log_prob(old_probs, old_actions)

        ratio = torch.exp(log_probs - old_logprobs)
        p1 = ratio * adv_value
        p2 = torch.clamp(ratio, 1.0 - self._epsilon, 1.0 + self._epsilon) * adv_value
        loss_policy = -torch.min(p1, p2)
        loss_policy = loss_policy.mean()

        entropy = self._network.actor.entropy(probs)
        loss = loss_value * self._critic_loss_weight + loss_policy * self._actor_loss_weight + self._beta * entropy

        return loss

    def calc_advantage(self, values, rewards, dones, gamma):
        buffer_size = rewards.shape[0]

        returns = torch.zeros((buffer_size, self._n_env, 1))
        advantages = torch.zeros((buffer_size, self._n_env, 1))

        last_gae = torch.zeros(self._n_env, 1)

        for n in reversed(range(buffer_size - 1)):
            delta = rewards[n] + dones[n] * gamma * values[n + 1] - values[n]
            last_gae = delta + dones[n] * gamma * self._lambda * last_gae

            returns[n] = last_gae + values[n]
            advantages[n] = last_gae

        return returns, advantages
