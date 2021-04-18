import time
from utils import *


class PPO:
    def __init__(self, network, lr, actor_loss_weight, critic_loss_weight, batch_size, trajectory_size, p_beta, p_gamma, log_prob_fn, entropy_fn,
                 ppo_epochs=10, p_epsilon=0.2, p_lambda=0.95, weight_decay=0, device='cpu', n_env=1):
        self._network = network
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr, weight_decay=weight_decay)
        self._beta = p_beta
        self._gamma = p_gamma
        self._epsilon = p_epsilon
        self._lambda = p_lambda
        self._device = device
        self._batch_size = batch_size
        self._trajectory_size = trajectory_size
        self._actor_loss_weight = actor_loss_weight
        self._critic_loss_weight = critic_loss_weight
        self._log_prob_fn = log_prob_fn
        self._entropy_fn = entropy_fn

        self._trajectory = []
        self._ppo_epochs = ppo_epochs
        self._motivation = None
        self._n_env = n_env

        if self._n_env > 1:
            self._trajectories = [[] for _ in range(self._n_env)]

    def train(self, state0, value, action, prob, state1, reward, done):
        self._trajectory.append((state0, value, action, prob, state1, reward, done))

        if len(self._trajectory) == self._trajectory_size:
            start = time.time()
            states = []
            values = []
            actions = []
            probs = []
            next_states = []
            rewards = []
            dones = []

            for state, value, action, prob, next_state, reward, done in self._trajectory:
                states.append(state)
                values.append(value)
                actions.append(action)
                probs.append(prob)
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)

            states = torch.stack(states).squeeze(1)
            values = torch.stack(values).squeeze(1)
            actions = torch.stack(actions).squeeze(1)
            probs = torch.stack(probs).squeeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self._device)
            dones = torch.tensor(dones, dtype=torch.float32, device=self._device)
            del self._trajectory[:]

            intrinsic_reward = None
            if self._motivation:
                next_states = torch.stack(next_states).squeeze(1)
                intrinsic_reward = self.calc_int_reward(states, actions, next_states)

            adv_values, ref_values = self.calc_advantage(values, rewards, dones, intrinsic_reward)
            self._train(states, next_states, actions, probs, adv_values, ref_values)

            end = time.time()
            print("Trajectory {0:d} batch size {1:d} epochs {2:d} training time {3:.2f}s".format(self._trajectory_size, self._batch_size, self._ppo_epochs, end - start))

    def train_n_env(self, state0, value, action, prob, state1, reward, done):
        trajectory_size_1_env = self._trajectory_size // self._n_env

        for i in range(self._n_env):
            self._trajectories[i].append((state0[i].unsqueeze(0), value[i].unsqueeze(0), action[i].unsqueeze(0), prob[i].unsqueeze(0), state1[i].unsqueeze(0), reward[i], done[i]))

        if len(self._trajectories[0]) == trajectory_size_1_env:
            start = time.time()

            states_t = []
            next_states_t = []
            actions_t = []
            probs_t = []
            adv_values_t = []
            ref_values_t = []

            for i in range(self._n_env):
                start = time.time()
                states = []
                values = []
                actions = []
                probs = []
                next_states = []
                rewards = []
                dones = []

                for state, value, action, prob, next_state, reward, done in self._trajectories[i]:
                    states.append(state)
                    values.append(value)
                    actions.append(action)
                    probs.append(prob)
                    next_states.append(next_state)
                    rewards.append(reward)
                    dones.append(done)

                states = torch.stack(states).squeeze(1)
                values = torch.stack(values).squeeze(1)
                actions = torch.stack(actions).squeeze(1)
                probs = torch.stack(probs).squeeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)
                self._trajectories[i].clear()

                intrinsic_reward = None
                if self._motivation:
                    next_states = torch.stack(next_states).squeeze(1)
                    next_states_t.append(next_states)
                    intrinsic_reward = self.calc_int_reward(states, actions, next_states)

                adv_values, ref_values = self.calc_advantage(values, rewards, dones, intrinsic_reward)

                states_t.append(states)
                actions_t.append(actions)
                probs_t.append(probs)
                adv_values_t.append(adv_values)
                ref_values_t.append(ref_values)

            states_t = torch.cat(states_t, dim=0).to(self._device)
            if len(next_states_t) > 0:
                next_states_t = torch.cat(next_states_t, dim=0).to(self._device)
            actions_t = torch.cat(actions_t, dim=0).to(self._device)
            probs_t = torch.cat(probs_t, dim=0).to(self._device)
            adv_values_t = torch.cat(adv_values_t, dim=0).to(self._device)
            ref_values_t = torch.cat(ref_values_t, dim=0).to(self._device)

            self._train(states_t, next_states_t, actions_t, probs_t, adv_values_t, ref_values_t)

            end = time.time()
            print("Trajectory {0:d} batch size {1:d} epochs {2:d} training time {3:.2f}s".format(self._trajectory_size, self._batch_size, self._ppo_epochs, end - start))

    def _train(self, states, next_states, actions, probs, adv_values, ref_values):
        adv_values = (adv_values - torch.mean(adv_values)) / torch.std(adv_values)

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
                probs_v = probs[batch_ofs:batch_l]
                batch_adv_v = adv_values[batch_ofs:batch_l].unsqueeze(-1)
                batch_ref_v = ref_values[batch_ofs:batch_l].unsqueeze(-1)

                self._optimizer.zero_grad()
                loss = self.calc_loss(states_v, batch_ref_v, batch_adv_v, actions_v, probs_v)
                loss.backward()
                self._optimizer.step()

    def calc_loss(self, states, ref_value, adv_value, old_actions, old_probs):
        values, _, probs = self._network(states)

        loss_value = torch.nn.functional.mse_loss(values, ref_value)

        log_probs = self._log_prob_fn(probs, old_actions)
        old_logprobs = self._log_prob_fn(old_probs, old_actions)

        ratio = torch.exp(log_probs - old_logprobs) * adv_value
        clipped_ratio = torch.clamp(ratio, 1.0 - self._epsilon, 1.0 + self._epsilon) * adv_value
        loss_policy = -torch.min(ratio, clipped_ratio)
        loss_policy = loss_policy.mean()

        entropy = self._entropy_fn(probs)
        loss = loss_value * self._critic_loss_weight + loss_policy * self._actor_loss_weight + self._beta * entropy

        return loss

    def calc_int_reward(self, states, actions, next_states):
        int_reward = self._motivation.reward(states[0:self._batch_size], actions[0:self._batch_size], next_states[0:self._batch_size]).detach()

        for batch_ofs in range(self._batch_size, self._trajectory_size, self._batch_size):
            batch_l = min(batch_ofs + self._batch_size, self._trajectory_size)
            int_reward = torch.cat([int_reward, self._motivation.reward(states[batch_ofs:batch_l], actions[batch_ofs:batch_l], next_states[batch_ofs:batch_l]).detach()], dim=0)
        int_reward = int_reward.squeeze()

        return int_reward

    def calc_advantage(self, values, rewards, dones, intrinsic_reward):
        dones = dones[:-1].flip(0)
        rewards = rewards[:-1].flip(0)
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

        adv_v = torch.tensor(adv_v, dtype=torch.float32).flip(0)
        ref_v = adv_v + val.flip(0)

        return adv_v, ref_v

    def add_motivation_module(self, motivation):
        self._motivation = motivation

    def get_motivation_module(self):
        return self._motivation
