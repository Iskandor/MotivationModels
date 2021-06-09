import numpy as np
import psutil
import torch
import umap
from scipy.cluster.vq import kmeans, whiten
from tqdm import tqdm

from modules.rnd_models.RNDModelBullet import *
from plots.video import VideoRenderer


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

    def initialization_test(self, config, n=1000):
        reducer = umap.UMAP(n_jobs=psutil.cpu_count(logical=True))
        grid_embedding = reducer.fit_transform(self.grid)
        renderer = VideoRenderer()

        models = [QRNDModelBullet1, QRNDModelBullet2]

        for index, m in enumerate(models):
            self.errors = []
            for i in tqdm(range(n)):
                self.motivation.network.motivator = m(2, 1, config)
                self.errors.append(self.motivation.error(self.grid_state, self.grid_action))
            self.errors = torch.stack(self.errors)
            errors = torch.sum(self.errors, dim=0)
            mean = torch.mean(errors, dim=0)
            std = torch.std(errors, dim=0)
            print('Mean: {0:f} Std: {1:f}'.format(mean, std))

            renderer.render_error(index, grid_embedding, errors.numpy())

    def collect(self, state, action):
        self.states.append(state.squeeze(0).numpy())
        self.actions.append(action.squeeze(0).numpy())
        if self.motivation is not None:
            self.errors.append(self.motivation.error(self.grid_state, self.grid_action))

    def end_trajectory(self, train_ext_reward):
        states = np.stack(self.states)
        actions = np.stack(self.actions)
        errors = torch.mean(torch.stack(self.errors), dim=0).numpy()

        if self.error_max < errors.max():
            self.error_max = errors.max()

        del self.states[:]
        del self.actions[:]

        trajectory = (states, actions, errors, train_ext_reward)
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
