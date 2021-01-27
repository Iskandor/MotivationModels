import time

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

    def train(self, state0, action, state1, reward, done):
        self._trajectory.append((state0, action, state1, reward, done))

        if len(self._trajectory) == self._trajectory_size:
            self._train()

    def train_n_env(self, state0, action, state1, reward, done):
        for i in range(self._n_env):
            self._trajectories[i].append((state0[i], action[i].unsqueeze(0), state1[i], reward[i], done[i]))

        if len(self._trajectories[0]) == self._trajectory_size // self._n_env:
            for i in range(self._n_env):
                self._trajectory += self._trajectories[i]
                self._trajectories[i].clear()

            self._train()

    def _train(self):
        start = time.time()

        states = []
        actions = []
        next_states = []
        for state, action, next_state, _, _ in self._trajectory:
            states.append(state)
            actions.append(action)
            next_states.append(next_state)

        states = torch.stack(states).squeeze(1)
        actions = torch.stack(actions).squeeze(1)

        intrinsic_reward = None
        if self._motivation:
            self._motivation.to('cpu')
            next_states = torch.stack(next_states).squeeze(1).to('cpu')
            actions_code = one_hot_code(actions, self._agent.action_dim, 'cpu')
            intrinsic_reward = self._motivation.reward(states, actions_code, next_states)
            self._motivation.to(self._device)
            actions_code = actions_code.to(self._device)
            next_states = next_states.to(self._device)

        traj_adv_v, traj_ref_v = self.calc_advantage(states, intrinsic_reward)
        traj_adv_v = (traj_adv_v - torch.mean(traj_adv_v)) / torch.std(traj_adv_v)
        old_logprob = self.calc_log_probs(states, actions)

        trajectory = self._trajectory[:-1]

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

                probs_v = torch.gather(self._agent.action(states_v), 1, actions_v)
                logprob_pi_v = probs_v.log()
                ratio_v = torch.exp(logprob_pi_v - batch_old_logprob_v)
                surr_obj_v = batch_adv_v * ratio_v
                c_ratio_v = torch.clamp(ratio_v, 1.0 - self._epsilon, 1.0 + self._epsilon)
                clipped_surr_v = batch_adv_v * c_ratio_v
                loss_policy_v = -torch.min(surr_obj_v, clipped_surr_v).mean()

                self._optimizer.zero_grad()
                loss = loss_value_v * self._critic_loss_weight + loss_policy_v * self._actor_loss_weight
                if self._motivation:
                    next_states_v = next_states[batch_ofs:batch_l]
                    actions_code_v = actions_code[batch_ofs:batch_l]
                    loss += self._motivation.loss(states_v, actions_code_v, next_states_v)
                loss.backward()
                self._optimizer.step()

        del self._trajectory[:]

        end = time.time()
        print("Trajectory {0:d} batch size {1:d} epochs {2:d} training time {3:.2f}s".format(self._trajectory_size, self._batch_size, self._ppo_epochs, end - start))

    def calc_log_probs(self, states, actions):
        probs = self._agent.action(states[0:self._batch_size]).detach()
        for batch_ofs in range(self._batch_size, self._trajectory_size, self._batch_size):
            batch_l = min(batch_ofs + self._batch_size, self._trajectory_size)
            probs = torch.cat([probs, self._agent.action(states[batch_ofs:batch_l]).detach()], dim=0)
        return torch.gather(probs, 1, actions)[:-1].log()

    def calc_advantage(self, states, intrinsic_reward):
        values = self._agent.value(states[0:self._batch_size]).detach()

        for batch_ofs in range(self._batch_size, self._trajectory_size, self._batch_size):
            batch_l = min(batch_ofs + self._batch_size, self._trajectory_size)
            values = torch.cat([values, self._agent.value(states[batch_ofs:batch_l]).detach()], dim=0)
        values = values.squeeze()

        last_gae = 0.0
        adv_v = torch.zeros(self._trajectory_size - 1, dtype=torch.float32, device=self._device)
        ref_v = torch.zeros(self._trajectory_size - 1, dtype=torch.float32, device=self._device)
        index = self._trajectory_size - 2

        if intrinsic_reward is None:
            for val, next_val, exp in zip(reversed(values[:-1]), reversed(values[1:]), reversed(self._trajectory[:-1])):
                if exp[4]:
                    delta = exp[3] - val
                    last_gae = delta
                else:
                    delta = exp[3] + self._gamma * next_val - val
                    last_gae = delta + self._gamma * self._lambda * last_gae

                adv_v[index] = last_gae
                ref_v[index] = last_gae + val
                index -= 1
        else:
            for val, next_val, exp, ir in zip(reversed(values[:-1]), reversed(values[1:]), reversed(self._trajectory[:-1]), reversed(intrinsic_reward[:-1])):
                if exp[4]:
                    delta = exp[3] + ir - val
                    last_gae = delta
                else:
                    delta = exp[3] + ir + self._gamma * next_val - val
                    last_gae = delta + self._gamma * self._lambda * last_gae

                adv_v[index] = last_gae
                ref_v[index] = last_gae + val
                index -= 1

        return adv_v, ref_v

    def get_action(self, state, deterministic=False):
        probs = self._agent.action(state)
        if deterministic:
            action = probs.argmax()
        else:
            m = Categorical(probs)
            action = m.sample()

        return action.unsqueeze(probs.ndim - 1)

    def add_motivation(self, motivation):
        self._motivation = motivation

    def get_motivation(self):
        return self._motivation

    def save(self, path):
        torch.save(self._agent.state_dict(), path + '.pth')

    def load(self, path):
        self._agent.load_state_dict(torch.load(path + '.pth'))
