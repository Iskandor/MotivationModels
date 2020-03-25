import abc

import torch


class MetaLearnerModel(torch.nn.Module):
    @abc.abstractmethod
    def __init__(self, state_dim, action_dim):
        super(MetaLearnerModel, self).__init__()

    @abc.abstractmethod
    def forward(self, state, action):
        raise NotImplementedError


class MetaLearnerMotivation:
    def __init__(self, network, forward_model, state_dim, action_dim, lr, weight_decay=0):
        self._forward_model = forward_model
        self._network = network(state_dim, action_dim)
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, state0, action, state1):
        self._optimizer.zero_grad()
        error = torch.tensor([self._forward_model.reward(state0, action, state1)], dtype=torch.float32)
        loss = torch.nn.functional.mse_loss(self._network(state0, action), error)
        loss.backward()
        self._optimizer.step()

        self._forward_model.train(state0, action.detach(), state1)

    def reward(self, state0, action, state1, eta=1.0):
        uncertainty = self._forward_model.reward(state0, action, state1)
        surprise = self._surprise_reward(state0, action, state1, 0.1)
        #print(str(uncertainty) + ' ' + str(surprise))
        return max(uncertainty, surprise)

    def _surprise_reward(self, state0, action, state1, eta=1.0):
        error = torch.tensor([self._forward_model.reward(state0, action, state1)], dtype=torch.float32)
        reward = (error / self._network(state0, action)) - 1
        return max(reward.item(), 0)
