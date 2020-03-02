import torch

from ReplayBuffer import ReplayBuffer, Transition


class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        hidden = 8

        self._hidden0 = torch.nn.Linear(state_dim, hidden)
        self._hidden1 = torch.nn.Linear(hidden + action_dim, int(hidden / 2))
        self._output = torch.nn.Linear(int(hidden / 2), 1)

        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.nn.functional.relu(self._hidden0(state))
        x = torch.cat([x, action], 1)
        x = torch.nn.functional.relu(self._hidden1(x))
        value = self._output(x)
        return value


class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        hidden = 8

        self._hidden0 = torch.nn.Linear(state_dim, hidden)
        self._hidden1 = torch.nn.Linear(hidden, int(hidden / 2))
        self._output = torch.nn.Linear(int(hidden / 2), action_dim)

        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-3, 3e-3)

    def forward(self, state):
        x = torch.nn.functional.relu(self._hidden0(state))
        x = torch.nn.functional.relu(self._hidden1(x))
        policy = torch.tanh(self._output(x))
        return policy


class DDPG:
    def __init__(self, state_dim, action_dim, memory_size, sample_size, actor_lr, critic_lr, gamma, tau, weight_decay=0):
        self._actor = Actor(state_dim, action_dim)
        self._critic = Critic(state_dim, action_dim)
        self._actor_target = Actor(state_dim, action_dim)
        self._critic_target = Critic(state_dim, action_dim)
        self._memory = ReplayBuffer(memory_size)
        self._sample_size = sample_size
        self._gamma = gamma
        self._tau = tau

        self._hard_update(self._actor_target, self._actor)
        self._hard_update(self._critic_target, self._critic)

        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=actor_lr)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=critic_lr, weight_decay=weight_decay)

    def get_action(self, state):
        return self._actor(state)

    def train(self, state0, action0, state1, reward, done):
        self._memory.add((state0, action0, state1, reward, done))
        sample = self._memory.sample(self._sample_size)

        if sample:
            s0, a0, s1, r, d = zip(*sample)

            batch_state0 = torch.cat(s0)
            batch_state1 = torch.cat(s1)
            batch_action0 = torch.cat(a0)
            batch_reward = torch.cat(r).unsqueeze(1)
            batch_done = torch.cat(d).unsqueeze(1)

            critic_target = batch_reward + batch_done * self._gamma * self._critic_target(batch_state1, self._actor_target(batch_state1))
            # self._critic(batch_state0, batch_action0)
            state = torch.zeros((4, 2), dtype=torch.float32)
            action = torch.zeros((4, 1), dtype=torch.float32)
            # self._critic(state, action)

            self._critic_optimizer.zero_grad()
            value_loss = torch.nn.functional.mse_loss(self._critic(batch_state0, batch_action0), critic_target.detach())
            value_loss.backward()
            self._critic_optimizer.step()

            self._actor_optimizer.zero_grad()
            policy_loss = -self._critic(batch_state0, self._actor(batch_state0))
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
