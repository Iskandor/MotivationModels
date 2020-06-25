import random
import gym
import gym_chess
import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical
from torchvision.models import ResNet

from algorithms.PPO import PPO


class Memory:
    def __init__(self):
        self.actions = []
        self.states0 = []
        self.states1 = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states0[:]
        del self.states1[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = torch.nn.functional.relu(out)
        return out


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = 2304

        self.backbone = nn.Sequential(
            nn.Conv2d(12, 64, 3, 1),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.Flatten()
        )

        # actor
        self.actor = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, int(self.feature_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(self.feature_dim / 2), action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, int(self.feature_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(self.feature_dim / 2), 1)
        )

        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_dim + action_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, constraint, device):
        state = state.reshape((1, 12, 8, 8)).to(device)
        constraint = torch.tensor(constraint, dtype=torch.int64, device=device)
        features = self.backbone(state)
        action_probs = self.actor(features).flatten()
        mask = torch.zeros_like(action_probs, dtype=torch.bool, device=device)
        mask = torch.scatter(mask, 0, constraint, 1)
        action_probs = action_probs * mask
        dist = Categorical(action_probs)
        action = dist.sample()

        return action.item(), action, dist.log_prob(action)

    def evaluate(self, state, action, device):
        features = self.backbone(state)
        action_probs = self.actor(features)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.critic(features)

        a = torch.zeros((action.shape[0], self.action_dim), dtype=torch.float32, device=device)
        a = torch.scatter(a, 1, action.reshape(action.shape[0], 1), 1)
        fm_input = torch.cat((features, a), dim=1)
        state_prediction = self.forward_model(fm_input)

        return action_logprobs, torch.squeeze(state_value), dist_entropy, state_prediction


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

    return torch.cat(inputs, 0)


def run_baseline(trials, episodes):
    env = gym.make('ChessVsRandomBot-v0')
    state_dim = 12 * env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = env.action_space.n

    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 4  # update policy for K epochs
    update_timestep = 2000
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
            steps = 0

            done = False
            while not done:
                steps += 1
                player = 1
                constraints = env.get_possible_actions(state, player)
                if len(constraints) == 0:
                    action = env.resign_action()
                    print('@' * 15, 'PLAYER RESIGNED', '@' * 15)
                    next_state, reward, done, info = env.step(action)
                else:
                    s0 = state_transform(state)
                    action, a, d = ppo.policy_old.act(s0, constraints, device)
                    if action not in constraints:
                        print(action)
                    next_state, reward, done, info = env.step(action)

                    s1 = state_transform(next_state)
                    timestep += 1
                    memory.states0.append(s0)
                    memory.states1.append(s1)
                    memory.actions.append(a)
                    memory.logprobs.append(d)
                    # Saving reward and is_terminal:
                    memory.rewards.append(reward)
                    memory.is_terminals.append(done)

                if done:
                    if reward < 0:
                        lose += 1
                    if reward > 0:
                        win += 1

                # update if its time
                if timestep % update_timestep == 0:
                    ppo.update(memory)
                    memory.clear_memory()
                    timestep = 0

                state = next_state

            print('Score: ' + str(win) + '/' + str(lose) + ' steps: ' + str(steps))
