import torch


class DDPG:
    def __init__(self, network, actor_lr, critic_lr, gamma, tau, motivation=None, device='cpu'):
        self.network = network
        self.motivation = motivation
        self._gamma = gamma
        self._tau = tau
        self.device = device

        self._critic_optimizer = torch.optim.Adam(self.network.critic.parameters(), lr=critic_lr)
        self._actor_optimizer = torch.optim.Adam(self.network.actor.parameters(), lr=actor_lr)

    def train_sample(self, memory, indices):
        if indices:
            sample = memory.sample(indices)

            states = sample.state.to(self.device)
            next_states = sample.next_state.to(self.device)
            actions = sample.action.to(self.device)
            rewards = sample.reward.to(self.device)
            masks = sample.mask.to(self.device)

            if self.motivation:
                if self.network.critic_heads == 1:
                    rewards += self.motivation.reward_sample(memory, indices)
                else:
                    ext_reward = rewards
                    int_reward = self.motivation.reward_sample(memory, indices)
                    rewards = (ext_reward, int_reward)

            self.train(states, actions, next_states, rewards, masks)

    def train(self, states, actions, next_states, rewards, masks):
        if self.network.critic_heads == 1:
            value = self.network.value(states, actions)
            value_target = self.network.value_target(next_states, self.network.action_target(next_states))

            expected_values = rewards + masks * self._gamma * value_target
            loss = torch.nn.functional.mse_loss(value, expected_values)

        else:
            value_ext, value_int = self.network.value(states, actions)
            value_target_ext, value_target_int = self.network.value_target(next_states, self.network.action_target(next_states))

            ext_reward, int_reward = rewards

            expected_values_ext = ext_reward + masks * self._gamma * value_target_ext
            expected_values_int = int_reward + masks * self._gamma * value_target_int
            loss = torch.nn.functional.mse_loss(value_ext, expected_values_ext) + torch.nn.functional.mse_loss(value_int, expected_values_int)

        self._critic_optimizer.zero_grad()
        loss.backward()
        self._critic_optimizer.step()

        if self.network.critic_heads == 1:
            actor_value = self.network.value(states, self.network.action(states))
        else:
            actor_value, _ = self.network.value(states, self.network.action(states))

        self._actor_optimizer.zero_grad()
        loss = -actor_value.mean()
        loss.backward()
        self._actor_optimizer.step()

        self.network.soft_update(self._tau)
