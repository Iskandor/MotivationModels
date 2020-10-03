import torch
from torch.distributions import Categorical


class A2C:
    def __init__(self, actor, critic, actor_lr, critic_lr, gamma, weight_decay=0):
        self._actor = actor
        self._critic = critic
        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=actor_lr)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=critic_lr, weight_decay=weight_decay)
        self._gamma = gamma

        self._trajectory = []
        self._rewards = []

    def train(self, reward, done):
        if done:
            R = 0
            policy_losses = []
            value_losses = []
            returns = []

            for r in self._rewards[::-1]:
                R = r + self._gamma * R
                returns.insert(0, R)

            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            for (log_prob, value), R in zip(self._trajectory, returns):
                advantage = R - value.item()
                policy_losses.append(-log_prob * advantage)
                value_losses.append(torch.nn.functional.smooth_l1_loss(value, torch.tensor([R])))

            self._actor_optimizer.zero_grad()
            policy_loss = torch.stack(policy_losses).sum()
            policy_loss.backward()
            self._actor_optimizer.step()

            self._critic_optimizer.zero_grad()
            value_loss = torch.stack(value_losses).sum()
            value_loss.backward()
            self._critic_optimizer.step()

            del self._rewards[:]
            del self._trajectory[:]
        else:
            self._rewards.append(reward)

    def get_action(self, state):
        probs = torch.softmax(self._actor(state), dim=-1)
        state_value = self._critic(state)

        m = Categorical(probs)
        action = m.sample()

        self._trajectory.append((m.log_prob(action), state_value))

        return action.item()