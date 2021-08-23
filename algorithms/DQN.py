import torch


class DQN:
    def __init__(self, network, critic_lr, gamma, weight_decay=0, motivation=None):
        self.network = network
        self.gamma = gamma
        self.update_step = 0
        self.motivation = motivation

        self.critic_optimizer = torch.optim.Adam(self.network.parameters(), lr=critic_lr, weight_decay=weight_decay)

    def train_sample(self, memory, indices):
        if indices:
            sample = memory.sample(indices)

            states = sample.state
            next_states = sample.next_state
            actions = sample.action
            rewards = sample.reward
            masks = sample.mask

            if self.motivation:
                rewards += self.motivation.reward_sample(memory, indices)

            self.train(states, actions, next_states, rewards, masks)

    def train(self, state0, action0, state1, reward, done):
            Qs0a = self.network.value(state0).gather(dim=1, index=action0.type(torch.int64)).squeeze()
            Qs1max = self.network.value_target(state1).max(1)[0]
            target = reward + done * self.gamma * Qs1max

            loss = torch.nn.functional.mse_loss(Qs0a, target.detach())
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            self.update_step += 1
            if self.update_step == 100:
                self.update_step = 0
                self.network.hard_update()
