import torch

from algorithms.DDPG import DDPGCritic, DDPGActor
from ddpg_experiment import ExperimentDDPG
from motivation.ForwardModelMotivation import ForwardModel
from motivation.MateLearnerMotivation import MetaLearnerModel


class Critic(DDPGCritic):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__(state_dim, action_dim)

        self._hidden0 = torch.nn.Linear(state_dim, 340)
        self._hidden1 = torch.nn.Linear(340 + action_dim, 220)
        self._output = torch.nn.Linear(220, 1)

        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.nn.functional.relu(self._hidden0(state))
        x = torch.cat([x, action], 1)
        x = torch.nn.functional.relu(self._hidden1(x))
        value = self._output(x)
        return value


class Actor(DDPGActor):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__(state_dim, action_dim)

        self._hidden0 = torch.nn.Linear(state_dim, 340)
        self._hidden1 = torch.nn.Linear(340, 220)
        self._output = torch.nn.Linear(220, action_dim)

        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-1, 3e-1)

    def forward(self, state):
        x = torch.nn.functional.relu(self._hidden0(state))
        x = torch.nn.functional.relu(self._hidden1(x))
        policy = torch.tanh(self._output(x))
        return policy


class ForwardModelNetwork(ForwardModel):
    def __init__(self, state_dim, action_dim):
        super(ForwardModelNetwork, self).__init__(state_dim, action_dim)
        self._hidden0 = torch.nn.Linear(state_dim + action_dim, 340)
        self._hidden1 = torch.nn.Linear(340, 340)
        self._output = torch.nn.Linear(340, state_dim)

        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-1, 3e-1)

    def forward(self, state, action):
        x = torch.cat([state, action], state.ndim - 1)
        x = torch.nn.functional.relu(self._hidden0(x))
        x = torch.nn.functional.relu(self._hidden1(x))
        value = self._output(x)
        return value


class MetaLearnerNetwork(MetaLearnerModel):
    def __init__(self, state_dim, action_dim):
        super(MetaLearnerNetwork, self).__init__(state_dim, action_dim)
        self._hidden0 = torch.nn.Linear(state_dim + action_dim, 340)
        self._hidden1 = torch.nn.Linear(340, 170)
        self._output = torch.nn.Linear(170, 1)

        torch.nn.init.xavier_uniform_(self._hidden0.weight)
        torch.nn.init.xavier_uniform_(self._hidden1.weight)
        torch.nn.init.uniform_(self._output.weight, -3e-1, 3e-1)

    def forward(self, state, action):
        x = torch.cat([state, action], state.ndim - 1)
        x = torch.nn.functional.relu(self._hidden0(x))
        x = torch.nn.functional.relu(self._hidden1(x))
        value = self._output(x)
        return value


def run_baseline(args):
    args.actor_lr = 1e-4
    args.critic_lr = 2e-4
    args.gamma = 0.99
    args.tau = 1e-3

    experiment = ExperimentDDPG('HalfCheetahPyBulletEnv-v0', Actor, Critic, ForwardModelNetwork, MetaLearnerNetwork)
    experiment.run_baseline(args)


def run_forward_model(args):
    args.actor_lr = 1e-4
    args.critic_lr = 2e-4
    args.gamma = 0.99
    args.tau = 1e-3
    args.forward_model_lr = 1e-3
    args.eta = 1

    experiment = ExperimentDDPG('HalfCheetahPyBulletEnv-v0', Actor, Critic, ForwardModelNetwork, MetaLearnerNetwork)
    experiment.run_forward_model(args)

def run_surprise_model(args):
    args.actor_lr = 1e-4
    args.critic_lr = 2e-4
    args.gamma = 0.99
    args.tau = 1e-3
    args.forward_model_lr = 2e-4
    args.metacritic_lr = 2e-3
    args.eta = 1
    args.metacritic_variant = 'C'

    experiment = ExperimentDDPG('HalfCheetahPyBulletEnv-v0', Actor, Critic, ForwardModelNetwork, MetaLearnerNetwork)
    experiment.run_metalearner_model(args)

def run_metalearner_model(args):
    args.actor_lr = 1e-4
    args.critic_lr = 2e-4
    args.gamma = 0.99
    args.tau = 1e-3
    args.forward_model_lr = 1e-3
    args.metacritic_lr = 2e-3
    args.eta = 1
    args.metacritic_variant = 'A'

    experiment = ExperimentDDPG('HalfCheetahPyBulletEnv-v0', Actor, Critic, ForwardModelNetwork, MetaLearnerNetwork)
    experiment.run_metalearner_model(args)
