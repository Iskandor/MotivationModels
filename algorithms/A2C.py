import torch
from torch.distributions import Categorical


class A2C:
    def __init__(self, agent, lr, actor_loss_weight, critic_loss_weight, gamma, batch_size, weight_decay=0, device='cpu'):
        self._agent = agent
        self._optimizer = torch.optim.Adam(self._agent.parameters(), lr=lr, weight_decay=weight_decay)
        self._beta = 0.1
        self._gamma = gamma
        self._device = device
        self._batch_size = batch_size
        self._actor_loss_weight = actor_loss_weight
        self._critic_loss_weight = critic_loss_weight

        self._trajectory = []
        self._rewards = []

    def train(self, state0, prob, log_prob, reward, done):
        state_value = self._agent.value(state0)

        self._trajectory.append((prob, log_prob, state_value))
        self._rewards.append(reward)

        if done or len(self._trajectory) == self._batch_size:
            R = 0
            policy_loss = []
            entropy_loss = []
            value_loss = []
            returns = []

            for r in self._rewards[::-1]:
                R = r + self._gamma * R
                returns.insert(0, R)

            returns = torch.tensor(returns, dtype=torch.float32).to(self._device)

            if len(self._trajectory) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            for (prob, log_prob, value), R in zip(self._trajectory, returns):
                advantage = R - value.item()
                policy_loss.append(-log_prob * advantage)
                entropy_loss.append(prob * log_prob)
                value_loss.append(torch.nn.functional.smooth_l1_loss(value, R.unsqueeze(0)))

            self._optimizer.zero_grad()
            entropy_loss = torch.stack(entropy_loss).sum(dim=1).mean()
            policy_loss = torch.stack(policy_loss).sum()
            value_loss = torch.stack(value_loss).sum()
            loss = policy_loss * self._actor_loss_weight + self._beta * entropy_loss + value_loss * self._critic_loss_weight
            loss.backward()
            self._optimizer.step()

            del self._rewards[:]
            del self._trajectory[:]

    def get_action(self, state):
        probs = torch.softmax(self._agent.action(state), dim=-1)
        m = Categorical(probs)
        action = m.sample()

        return action.item(), probs, m.log_prob(action)

    def get_probs(self, state):
        probs = torch.softmax(self._agent.action(state), dim=-1)

        return probs.detach()

    def save(self, path):
        torch.save(self._agent.state_dict(), path + '.pth')

    def load(self, path):
        self._agent.load_state_dict(torch.load(path + '.pth'))
