import time

from torch import nn
from torch.distributions import Categorical

from utils import *


class PPO:
    def __init__(self, agent, lr, actor_loss_weight, critic_loss_weight, batch_size, trajectory_size, p_beta, p_gamma, p_epsilon=0.2, p_lambda=0.95,
                 weight_decay=0, device='cpu', n_env=1):
        self._agent = agent
        self._optimizer = torch.optim.Adam(self._agent.parameters(), lr=lr, weight_decay=weight_decay)
        self._beta = p_beta
        self._gamma = p_gamma
        self._epsilon = p_epsilon
        self._lambda = p_lambda
        self._device = device
        self._batch_size = batch_size
        self._trajectory_size = trajectory_size
        self._actor_loss_weight = actor_loss_weight
        self._critic_loss_weight = critic_loss_weight
        self._trajectory = []
        self._ppo_epochs = 10
        self._motivation = None
        self._n_env = n_env

        if self._n_env > 1:
            self._trajectories = [[] for _ in range(self._n_env)]

    def action_dim(self):
        return self._agent.action_dim

    def train(self, state0, action, log_prob, state1, reward, done):
        self._trajectory.append((state0, action, log_prob, state1, reward, done))

        if len(self._trajectory) == self._trajectory_size:
            self._train()

    def train_n_env(self, state0, action, log_prob, state1, reward, done):
        for i in range(self._n_env):
            self._trajectories[i].append((state0[i], action[i], log_prob[i], state1[i], reward[i], done[i]))

        if len(self._trajectories[0]) == self._trajectory_size // self._n_env:
            for i in range(self._n_env):
                self._trajectory += self._trajectories[i]
                self._trajectories[i].clear()

            self._train()

    def _train(self):
        start = time.time()

        states = []
        actions = []
        old_logprob = []
        next_states = []
        rewards = []
        dones = []

        for state, action, log_prob, next_state, reward, done in self._trajectory:
            states.append(state)
            actions.append(action)
            old_logprob.append(log_prob)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)

        states = torch.stack(states).squeeze(1)
        actions = torch.stack(actions).squeeze(1)
        old_logprob = torch.stack(old_logprob).squeeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self._device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self._device)

        intrinsic_reward = None
        if self._motivation:
            next_states = torch.stack(next_states).squeeze(1)
            intrinsic_reward = self.calc_int_reward(states, actions, next_states)

        traj_adv_v, traj_ref_v = self.calc_advantage(states, rewards, dones, intrinsic_reward)
        traj_adv_v = (traj_adv_v - torch.mean(traj_adv_v)) / torch.std(traj_adv_v)

        trajectory = self._trajectory[:-1]

        if self._motivation:
            for batch_ofs in range(0, len(trajectory), self._batch_size):
                batch_l = min(batch_ofs + self._batch_size, len(trajectory))
                states_v = states[batch_ofs:batch_l]
                actions_v = actions[batch_ofs:batch_l]
                next_states_v = next_states[batch_ofs:batch_l]
                self._motivation.train(states_v, actions_v, next_states_v)

        for epoch in range(self._ppo_epochs):
            for batch_ofs in range(0, len(trajectory), self._batch_size):
                batch_l = min(batch_ofs + self._batch_size, len(trajectory))
                states_v = states[batch_ofs:batch_l]
                actions_v = actions[batch_ofs:batch_l]
                batch_adv_v = traj_adv_v[batch_ofs:batch_l].unsqueeze(-1)
                batch_ref_v = traj_ref_v[batch_ofs:batch_l]
                batch_old_logprob_v = old_logprob[batch_ofs:batch_l]

                value_v = self._agent.value(states_v)
                loss_value_v = torch.nn.functional.mse_loss(value_v.squeeze(-1), batch_ref_v)

                log_probs_v, entropy_v = self._agent.evaluate(states_v, actions_v)

                ratio_v = torch.exp(log_probs_v - batch_old_logprob_v)
                surr_obj_v = batch_adv_v * ratio_v
                c_ratio_v = torch.clamp(ratio_v, 1.0 - self._epsilon, 1.0 + self._epsilon)
                clipped_surr_v = batch_adv_v * c_ratio_v
                loss_policy_v = -torch.min(surr_obj_v, clipped_surr_v).mean()

                self._optimizer.zero_grad()
                loss = loss_value_v * self._critic_loss_weight + loss_policy_v * self._actor_loss_weight + self._beta * entropy_v
                loss.backward()
                self._optimizer.step()

        del self._trajectory[:]

        end = time.time()
        print("Trajectory {0:d} batch size {1:d} epochs {2:d} training time {3:.2f}s".format(self._trajectory_size, self._batch_size, self._ppo_epochs, end - start))

    def calc_int_reward(self, states, actions, next_states):
        int_reward = self._motivation.reward(states[0:self._batch_size], actions[0:self._batch_size], next_states[0:self._batch_size]).detach()

        for batch_ofs in range(self._batch_size, self._trajectory_size, self._batch_size):
            batch_l = min(batch_ofs + self._batch_size, self._trajectory_size)
            int_reward = torch.cat([int_reward, self._motivation.reward(states[batch_ofs:batch_l], actions[batch_ofs:batch_l], next_states[batch_ofs:batch_l]).detach()], dim=0)
        int_reward = int_reward.squeeze()

        return int_reward

    def calc_advantage(self, states, rewards, dones, intrinsic_reward):
        dones = dones[:-1].flip(0)
        rewards = rewards[:-1].flip(0)
        values = self._agent.value(states[0:self._batch_size]).detach()

        for batch_ofs in range(self._batch_size, self._trajectory_size, self._batch_size):
            batch_l = min(batch_ofs + self._batch_size, self._trajectory_size)
            values = torch.cat([values, self._agent.value(states[batch_ofs:batch_l]).detach()], dim=0)
        values = values.squeeze()

        val = values[:-1].flip(0)
        next_val = values[1:].flip(0) * self._gamma * dones
        delta = rewards + next_val - val
        gamma_lambda = self._gamma * self._lambda * dones

        if intrinsic_reward is not None:
            delta += intrinsic_reward[:-1].flip(0)

        last_gae = 0.0
        adv_v = []

        for d, gl in zip(delta, gamma_lambda):
            last_gae = d + gl * last_gae
            adv_v.append(last_gae)

        adv_v = torch.tensor(adv_v, dtype=torch.float32, device=self._device).flip(0)
        ref_v = adv_v + val.flip(0)

        return adv_v, ref_v

    def get_action(self, state, deterministic=False):
        return self._agent.action(state, deterministic)

    def add_motivation_module(self, motivation):
        self._motivation = motivation

    def get_motivation_module(self):
        return self._motivation

    def save(self, path):
        torch.save(self._agent.state_dict(), path + '.pth')

    def load(self, path):
        self._agent.load_state_dict(torch.load(path + '.pth'))
