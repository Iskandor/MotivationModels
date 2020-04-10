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
        error = self._forward_model.error(state0, action, state1)
        loss = torch.nn.functional.mse_loss(self._network(state0, action), error)
        loss.backward()
        self._optimizer.step()

        self._forward_model.train(state0, action.detach(), state1)

    def error(self, state0, action):
        return self._network(state0, action)

    def reward(self, state0, action, state1, eta=1.0):
        uncertainty = self._forward_model.reward(state0, action, state1)
        surprise = self._surprise_reward(state0, action, state1)
        #print(str(uncertainty) + ' ' + str(surprise))
        reward = None
        if state0.ndim == 1:
            reward = max(uncertainty, surprise)
        if state0.ndim == 2:
            reward = torch.max(uncertainty, surprise)
        return reward * eta

    def _surprise_reward(self, state0, action, state1, eta=1.0):
        reward = None
        if state0.ndim == 1:
            error = self._forward_model.error(state0, action, state1)
            prediction = self._network(state0, action)
            torch.abs(error - prediction)
            reward = torch.exp(1 * torch.abs(error - prediction)).item() - 1
            '''
            if abs(error - prediction) > 0.01:
                reward = error / self._network(state0, action) + self._network(state0, action) / error - 2
                reward = max(torch.tanh(reward).item(), 0)
            else:
                reward = 0
            '''

        if state0.ndim == 2:
            reward = torch.zeros((state0.shape[0], 1))
            error = self._forward_model.error(state0, action, state1)
            prediction = self._network(state0, action)
            reward = torch.exp(1 * torch.abs(error - prediction)) - 1
            '''
            for i in range(state0.shape[0]):
                if abs(error[i] - prediction[i]) > 0.01:
                    reward[i] = max(torch.tanh(error[i] / prediction[i] + prediction[i] / error[i] - 2).item(), 0)
                else:
                    reward[i] = 0
             '''

        return reward * eta
