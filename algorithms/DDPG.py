import torch


class DDPG:
    def __init__(self, network, actor_lr, critic_lr, gamma, tau, memory_buffer, sample_size, motivation=None):
        self.network = network
        self.motivation = motivation
        self._memory = memory_buffer
        self._sample_size = sample_size
        self._gamma = gamma
        self._tau = tau

        self._critic_optimizer = torch.optim.Adam(self.network.critic.parameters(), lr=critic_lr)
        self._actor_optimizer = torch.optim.Adam(self.network.actor.parameters(), lr=actor_lr)

    def train_sample(self, indices):
        if indices:
            sample = self._memory.sample(indices)

            states = sample.state
            next_states = sample.next_state
            actions = sample.action
            rewards = sample.reward
            masks = sample.mask

            if self.motivation:
                rewards += self.motivation.reward_sample(indices)

            self.train(states, actions, next_states, rewards, masks)

    def train(self, states, actions, next_states, rewards, masks):
        expected_values = rewards + masks * self._gamma * self.network.value_target(next_states, self.network.action_target(next_states))

        self._critic_optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(self.network.value(states, actions), expected_values)
        loss.backward()
        self._critic_optimizer.step()

        self._actor_optimizer.zero_grad()
        loss = -self.network.value(states, self.network.action(states)).mean()
        loss.backward()
        self._actor_optimizer.step()

        self.network.soft_update(self._tau)
