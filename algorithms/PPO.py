import time
from utils import *


class PPO:
    def __init__(self, network, lr, actor_loss_weight, critic_loss_weight, batch_size, trajectory_size, memory, p_beta, p_gamma,
                 ppo_epochs=10, p_epsilon=0.2, p_lambda=0.95, weight_decay=0, device='cpu', n_env=1, motivation=None):
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
            sample = self._memory.sample(indices)

            states_t = []
            actions_t = []
            probs_t = []
            adv_values_t = []
            ref_values_t = []

            trajectory_size_1_env = self._trajectory_size // self._n_env

            for i in range(self._n_env):
                start_batch = i * trajectory_size_1_env
                end_batch = (i + 1) * trajectory_size_1_env

                states = torch.stack(sample.state[start_batch:end_batch])
                values = torch.stack(sample.value[start_batch:end_batch])
                actions = torch.stack(sample.action[start_batch:end_batch])
                probs = torch.stack(sample.prob[start_batch:end_batch])
                rewards = torch.stack(sample.reward[start_batch:end_batch])
                dones = torch.stack(sample.mask[start_batch:end_batch])

                if self._motivation:
                    ext_adv_values, ext_ref_values = self.calc_advantage(values[:, 0], rewards[:, 0], dones.squeeze())
                    int_adv_values, int_ref_values = self.calc_advantage(values[:, 1], rewards[:, 1], dones.squeeze())
                    adv_values = ext_adv_values + int_adv_values
                    ref_values = torch.stack([ext_ref_values, int_ref_values], dim=1)
                else:
                    adv_values, ref_values = self.calc_advantage(values.squeeze(), rewards.squeeze(), dones.squeeze())

                states_t.append(states[:-1])
                actions_t.append(actions[:-1])
                probs_t.append(probs[:-1])
                adv_values_t.append(adv_values)
                ref_values_t.append(ref_values)

            states_t = torch.cat(states_t, dim=0).to(self._device)
            actions_t = torch.cat(actions_t, dim=0).to(self._device)
            probs_t = torch.cat(probs_t, dim=0).to(self._device)
            adv_values_t = torch.cat(adv_values_t, dim=0).to(self._device)
            ref_values_t = torch.cat(ref_values_t, dim=0).to(self._device)

            self._train(states_t, actions_t, probs_t, adv_values_t, ref_values_t)

            end = time.time()
            print("Trajectory {0:d} batch size {1:d} epochs {2:d} training time {3:.2f}s".format(self._trajectory_size, self._batch_size, self._ppo_epochs, end - start))

    def _train(self, states, actions, probs, adv_values, ref_values):
        if self._motivation is None:
            adv_values = (adv_values - torch.mean(adv_values)) / torch.std(adv_values)

        trajectory_size = self._trajectory_size - self._n_env

        for epoch in range(self._ppo_epochs):
            for batch_ofs in range(0, trajectory_size, self._batch_size):
                batch_l = min(batch_ofs + self._batch_size, trajectory_size)
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

        if self._motivation:
            ref_value = ref_value.squeeze(-1)
            loss_ext_value = torch.nn.functional.mse_loss(values[:, 0], ref_value[:, 0])
            loss_int_value = torch.nn.functional.mse_loss(values[:, 1], ref_value[:, 1])
            loss_value = loss_ext_value + loss_int_value
        else:
            loss_value = torch.nn.functional.mse_loss(values, ref_value)

        log_probs = self._network.actor.log_prob(probs, old_actions)
        old_logprobs = self._network.actor.log_prob(old_probs, old_actions)

        ratio = torch.exp(log_probs - old_logprobs) * adv_value
        clipped_ratio = torch.clamp(ratio, 1.0 - self._epsilon, 1.0 + self._epsilon) * adv_value
        loss_policy = -torch.min(ratio, clipped_ratio)
        loss_policy = loss_policy.mean()

        entropy = self._network.actor.entropy(probs)
        loss = loss_value * self._critic_loss_weight + loss_policy * self._actor_loss_weight + self._beta * entropy

        return loss

    def calc_int_reward(self, states, actions, next_states):
        int_reward = self._motivation.reward(states, actions, next_states)
        return int_reward

    def calc_advantage(self, values, rewards, dones):
        dones = dones[:-1].flip(0)
        rewards = rewards[:-1].flip(0)

        val = values[:-1].flip(0)
        next_val = values[1:].flip(0) * self._gamma * dones
        delta = rewards + next_val - val
        gamma_lambda = self._gamma * self._lambda * dones

        last_gae = 0.0
        adv_v = []

        for d, gl in zip(delta, gamma_lambda):
            last_gae = d + gl * last_gae
            adv_v.append(last_gae)

        adv_v = torch.tensor(adv_v, dtype=torch.float32, device=values.device).flip(0)
        ref_v = adv_v + val.flip(0)

        return adv_v, ref_v
