import torch
from torch.distributions import Categorical

from utils import *


class PPO:
    def __init__(self, agent, lr, actor_loss_weight, critic_loss_weight, batch_size, trajectory_size, p_beta, p_gamma, p_epsilon=0.2, p_lambda=0.95,
                 weight_decay=0, device='cpu'):
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

    def action_dim(self):
        return self._agent.action_dim

    def train(self, state0, action, state1, reward, done):
        self._trajectory.append((state0, action, state1, reward, done))

        if len(self._trajectory) == self._trajectory_size:
            states = []
            actions = []
            next_states = []
            for state, action, next_state, _, _ in self._trajectory:
                states.append(state)
                actions.append(action)
                next_states.append(next_state)

            states = torch.stack(states).squeeze(1).to('cpu')
            actions = torch.stack(actions).squeeze(1).to('cpu')

            intrinsic_reward = None
            if self._motivation:
                self._motivation.to('cpu')
                next_states = torch.stack(next_states).squeeze(1).to('cpu')
                actions_code = one_hot_code(actions, self._agent.action_dim, 'cpu')
                intrinsic_reward = self._motivation.reward(states, actions_code, next_states)
                self._motivation.to(self._device)
                actions_code = actions_code.to(self._device)
                next_states = next_states.to(self._device)

            self._agent.to('cpu')
            traj_adv_v, traj_ref_v = self.calc_advantage(states, intrinsic_reward)
            traj_adv_v = (traj_adv_v - torch.mean(traj_adv_v)) / torch.std(traj_adv_v)
            probs = torch.gather(torch.softmax(self._agent.action(states), dim=-1), 1, actions)
            self._agent.to(self._device)
            actions = actions.to(self._device)
            states = states.to(self._device)

            old_logprob = probs[:-1].log().detach().to(self._device)
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

                    probs_v = torch.gather(torch.softmax(self._agent.action(states_v), dim=-1), 1, actions_v)
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

    def calc_advantage(self, states, intrinsic_reward):
        values = self._agent.value(states)
        values = values.squeeze().data.cpu().numpy()

        last_gae = 0.0
        result_adv = []
        result_ref = []

        if intrinsic_reward is None:
            for val, next_val, exp in zip(reversed(values[:-1]), reversed(values[1:]), reversed(self._trajectory[:-1])):
                if exp[4]:
                    delta = exp[3] - val
                    last_gae = delta
                else:
                    delta = exp[3] + self._gamma * next_val - val
                    last_gae = delta + self._gamma * self._lambda * last_gae

                result_adv.append(last_gae)
                result_ref.append(last_gae + val)
        else:
            for val, next_val, exp, ir in zip(reversed(values[:-1]), reversed(values[1:]), reversed(self._trajectory[:-1]), reversed(intrinsic_reward[:-1])):
                if exp[4]:
                    delta = exp[3] + ir - val
                    last_gae = delta
                else:
                    delta = exp[3] + ir + self._gamma * next_val - val
                    last_gae = delta + self._gamma * self._lambda * last_gae

                result_adv.append(last_gae)
                result_ref.append(last_gae + val)

        adv_v = torch.FloatTensor(list(reversed(result_adv)))
        ref_v = torch.FloatTensor(list(reversed(result_ref)))
        return adv_v.to(self._device), ref_v.to(self._device)

    def get_action(self, state, deterministic=False):
        probs = torch.softmax(self._agent.action(state), dim=-1)
        if deterministic:
            action = probs.argmax()
        else:
            m = Categorical(probs)
            action = m.sample()

        return action.unsqueeze(0)

    def add_motivation(self, motivation):
        self._motivation = motivation

    def get_motivation(self):
        return self._motivation

    def save(self, path):
        torch.save(self._agent.state_dict(), path + '.pth')

    def load(self, path):
        self._agent.load_state_dict(torch.load(path + '.pth'))
