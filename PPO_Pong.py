from skimage import transform
import numpy as np
import gym
import torch
from etaprogress.progress import ProgressBar
from torch import nn
from torch.distributions import Categorical

from algorithms.PPO import PPO, Memory


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = 4096

        self.backbone = nn.Sequential(
            nn.Conv2d(state_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
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

    def forward(self):
        raise NotImplementedError

    def act(self, state, device):
        with torch.no_grad():
            state = state.reshape((1, state.shape[0], state.shape[1], state.shape[2])).to(device)
            features = self.backbone(state)
            action_probs = self.actor(features).flatten()
            dist = Categorical(action_probs)
            action = dist.sample()

        return action, dist.log_prob(action)

    def evaluate(self, state, action):
        features = self.backbone(state)
        action_probs = self.actor(features)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.critic(features)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


def scale_lumininance(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])


def preprocess_observations(obs):
    obs_gray = scale_lumininance(obs)
    obs_trans = transform.resize(obs_gray, (96, 96)).reshape(1, 96, 96)
    return obs_trans


def test(env, agent, device):
    state0 = torch.tensor(preprocess_observations(env.reset()), dtype=torch.float32)
    done = False
    test_reward = 0

    while not done:
        action0, log_prob = agent.policy_old.act(state0.clone().detach(), device)
        next_state, reward, done, _ = env.step(action0)
        test_reward += reward

    return test_reward


def run_baseline(args):
    env = gym.make('Pong-v0')
    lr = 0.001
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 5  # update policy for K epochs
    update_timestep = 5000
    eps_clip = 0.1
    device = 'cuda:0'

    for i in range(args.trials):
        timestep = 0
        rewards = np.zeros(args.episodes)
        state_dim = 1
        action_dim = 2

        memory = Memory()
        agent = PPO(ActorCritic, state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip, device)

        if args.load:
            agent.load(args.load)

        bar = ProgressBar(args.episodes, max_width=40)

        for e in range(args.episodes):
            state0 = torch.tensor(preprocess_observations(env.reset()), dtype=torch.float32)
            done = False
            train_reward = 0
            bar.numerator = e

            if e % 1000 == 0:
                torch.save(agent.policy, './models/pong_agent_' + str(e) + '.pth')

            while not done:
                env.render()
                action0, log_prob = agent.policy_old.act(state0.clone().detach(), device)
                next_state, reward, done, _ = env.step(action0 + 2)
                train_reward += reward
                state1 = torch.tensor(preprocess_observations(next_state), dtype=torch.float32)

                timestep += 1
                memory.states0.append(state0)
                memory.actions.append(action0)
                memory.logprobs.append(log_prob)
                memory.rewards.append(reward)
                memory.is_terminals.append(done)

                # update if its time
                if timestep % update_timestep == 0:
                    agent.update(memory)
                    memory.clear_memory()
                    timestep = 0

                state0 = state1

            print('Episode ' + str(e) + ' train reward ' + str(train_reward))
            print(bar)

        np.save('ppo_baseline_' + str(i), rewards)
