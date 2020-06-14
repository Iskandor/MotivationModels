import gym
import gym_chess
import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical

from algorithms.PPO import PPO


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(12, 16, (3, 3), padding=1),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3)),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(),
            nn.Flatten()
        )

        # actor
        self.actor = nn.Sequential(
            nn.Linear(32, 400),
            nn.Tanh(),
            nn.Linear(400, 400),
            nn.Tanh(),
            nn.Linear(400, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(32, 400),
            nn.Tanh(),
            nn.Linear(400, 400),
            nn.Tanh(),
            nn.Linear(400, 1),
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, constraint, device):
        state = state.to(device)
        constraint = torch.tensor(constraint, dtype=torch.int64, device=device)
        features = self.backbone(state)
        action_probs = self.actor(features).flatten()
        mask = torch.zeros_like(action_probs, dtype=torch.bool, device=device)
        mask = torch.scatter(mask, 0, constraint, 1)
        action_probs = action_probs * mask
        dist = Categorical(action_probs)
        action = dist.sample()

        return action.item(), state, action, dist.log_prob(action)

    def evaluate(self, state, action):
        features = self.backbone(state)
        action_probs = self.actor(features)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.critic(features)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


def find_pieces(board, ids, player):
    result = []
    for id in ids:
        index = np.where(board == id * player)
        if index[0].size > 0:
            result.append(index[0][0])
    return torch.tensor(result)


def state_transform(state):
    board = np.reshape(state['board'], 64)

    inputs = []

    for piece in [[9, 10, 11, 12, 13, 14, 15, 16], [1, 8], [2, 7], [3, 6], [4], [5]]:
        piece_board = torch.zeros(64, dtype=torch.float32)
        piece_index = find_pieces(board, piece, 1)
        if piece_index.shape[0] > 0:
            piece_board = torch.scatter(piece_board, 0, piece_index, 1)
        inputs.append(piece_board.reshape((1, 8, 8)))

    for piece in [[9, 10, 11, 12, 13, 14, 15, 16], [1, 8], [2, 7], [3, 6], [4], [5]]:
        piece_board = torch.zeros(64, dtype=torch.float32)
        piece_index = find_pieces(board, piece, -1)
        if piece_index.shape[0] > 0:
            piece_board = torch.scatter(piece_board, 0, piece_index, 1)
        inputs.append(piece_board.reshape((1, 8, 8)))

    return torch.cat(inputs, 0).reshape(1, 12, 8, 8)


def run_baseline(trials, episodes):
    env = gym.make('ChessVsRandomBot-v0')
    state_dim = 12 * env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n

    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 4  # update policy for K epochs
    update_timestep = 1000
    eps_clip = 0.2

    device = 'cuda:0'

    memory = Memory()
    ppo = PPO(ActorCritic, state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip, device)

    for i in range(trials):
        timestep = 0
        win = 0
        lose = 0
        for e in range(episodes):
            if e % 100 == 0:
                torch.save(ppo.policy, './models/chess_agent_' + str(e))

            state = env.reset()

            done = False
            while not done:
                player = 1
                constraints = env.get_possible_actions(state, player)
                state = state_transform(state)
                action, s, a, d = ppo.policy_old.act(torch.tensor(state, dtype=torch.float32), constraints, device)
                next_state, reward, done, info = env.step(action)

                timestep += 1
                memory.states.append(s)
                memory.actions.append(a)
                memory.logprobs.append(d)

                if not done:
                    # white
                    action = env.uniform_random_action()
                    next_state, reward, done, info = env.step(action)

                # Saving reward and is_terminal:
                memory.rewards.append(reward)
                memory.is_terminals.append(done)

                # update if its time
                if timestep % update_timestep == 0:
                    ppo.update(memory)
                    memory.clear_memory()
                    timestep = 0

                state = next_state

                if done:
                    if reward == -1:
                        lose += 1
                    if reward == 1:
                        win += 1
            print('Score: ' + str(win) + '/' + str(lose))
