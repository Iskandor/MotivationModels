import gym
import torch
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

        # actor
        self.action_layer = nn.Sequential(
            nn.Conv2d(6, 16, (5, 5), padding=2),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (5, 5)),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 400),
            nn.Tanh(),
            nn.Linear(400, 400),
            nn.Tanh(),
            nn.Linear(400, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Conv2d(6, 16, (5, 5), padding=2),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (5, 5)),
            nn.MaxPool2d((2, 2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 400),
            nn.Tanh(),
            nn.Linear(400, 400),
            nn.Tanh(),
            nn.Linear(400, 1),
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, device):
        state = state.to(device)
        action_probs = self.action_layer(state.reshape((1, 6, 19, 19)))
        dist = Categorical(action_probs)
        action = dist.sample()

        return action.item(), state, action, dist.log_prob(action)

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


def run_baseline(trials, episodes, batch_size, memory_size):
    env = gym.make('gym_go:go-v0', size=19, reward_method='real')
    # creating environment
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
    action_dim = env.action_space.n
    render = False
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
                torch.save(ppo.policy, './models/go_agent_' + str(e))

            state = env.reset()

            done = False
            while not done:
                #black
                action, s, a, d = ppo.policy_old.act(torch.tensor(state, dtype=torch.float32), device)

                try:
                    next_state, reward, done, info = env.step(action)
                except Exception as e:
                    continue

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
