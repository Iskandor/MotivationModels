import torch


class ICM:
    def __init__(self, model, beta=0.2, eta=1):
        self._model = model
        self._beta = beta
        self._eta = eta

    def loss(self, state0, action, state1):
        state_estimate = self._model.estimate_state(state0, action)
        state_target = self._model.get_features(state1).detach()
        loss_fm = torch.nn.functional.mse_loss(state_estimate, state_target)

        action_estimate = self._model.estimate_action(state0, state1)
        loss_im = torch.nn.functional.mse_loss(action_estimate, action)

        return (1 - self._beta) * loss_im + self._beta * loss_fm

    def error(self, state0, action, state1):
        with torch.no_grad():
            prediction = self._model.estimate_state(state0, action).detach()
            target = self._model.get_features(state1).detach()
            dim = len(prediction.shape) - 1
            error = torch.mean(torch.pow(prediction - target, 2), dim=dim).unsqueeze(dim)

        return error

    def reward(self, state0=None, action=None, state1=None, error=None):
        if error is None:
            reward = torch.tanh(self.error(state0, action, state1))
        else:
            reward = torch.tanh(error)

        return reward * self._eta

    def to(self, device):
        self._model.to(device)

    def save(self, path):
        torch.save(self._model.state_dict(), path + '_fm.pth')

    def load(self, path):
        self._model.load_state_dict(torch.load(path + '_fm.pth'))