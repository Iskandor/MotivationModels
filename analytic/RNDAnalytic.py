import numpy as np
import torch
from scipy.cluster.vq import kmeans, whiten


class RNDAnalytic:
    def __init__(self, config, motivation=None, grid=None):
        self.trajectory = []
        self.states = []
        self.actions = []
        self.errors = []
        self.config = config
        self.motivation = motivation
        self.error_max = 0.

        if grid is None:
            self.grid = None
        else:
            self.grid = torch.tensor(np.load(grid), dtype=torch.float32)
            self.grid_state = self.grid[:, :2]
            self.grid_action = self.grid[:, 2:]

    def collect(self, state, action):
        self.states.append(state.squeeze(0).numpy())
        self.actions.append(action.squeeze(0).numpy())
        if self.motivation is not None:
            self.errors.append(self.motivation.error(self.grid_state, self.grid_action))

    def end_trajectory(self):
        states = np.stack(self.states)
        actions = np.stack(self.actions)
        errors = torch.sum(torch.stack(self.errors), dim=0).numpy()

        if self.error_max < errors.max():
            self.error_max = errors.max()

        del self.states[:]
        del self.actions[:]

        trajectory = (states, actions, errors)
        self.trajectory.append(trajectory)

    def save_data(self, filename):
        data = {
            'grid_embedding': self.grid.numpy(),
            'trajectories': self.trajectory,
            'error_max': self.error_max
        }
        np.save('analytic_{0}'.format(filename), data)

    def generate_grid(self, trial, k=1000):
        self.states = np.stack(self.states)
        self.actions = np.stack(self.actions)

        grid, _ = kmeans(whiten(np.concatenate([self.states, self.actions], axis=1)), k)

        np.save('grid_{0}_{1}_{2}'.format(self.config.name, self.config.model, trial), grid)
