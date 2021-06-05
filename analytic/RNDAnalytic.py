import imageio
import numpy as np
import torch
import umap
from etaprogress.progress import ProgressBar
from scipy.cluster.vq import kmeans, whiten
import matplotlib.pyplot as plt


class RNDAnalytic:
    def __init__(self, config, motivation=None, grid=None):
        self.images = []
        self.trajectory = []
        self.states = []
        self.actions = []
        self.errors = []
        self.config = config
        self.motivation = motivation
        self.error_min = 0.
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
        states = torch.tensor(np.stack(self.states), dtype=torch.float32, device='cpu')
        actions = torch.tensor(np.stack(self.actions), dtype=torch.float32, device='cpu')
        errors = torch.sum(torch.stack(self.errors), dim=0).numpy()
        if self.error_min > errors.min():
            self.error_min = errors.min()
        if self.error_max < errors.max():
            self.error_max = errors.max()

        trajectory = (states, actions, errors)
        self.trajectory.append(trajectory)

    def render_video(self, filename):
        print('Rendering video {0:s}.mp4'.format(filename))
        reducer = umap.UMAP()
        grid_embedding = reducer.fit_transform(self.grid.numpy())
        bar = ProgressBar(len(self.trajectory), max_width=40)
        for i, t in enumerate(self.trajectory):
            states, actions, errors = t
            self.render_frame(i, reducer, grid_embedding, states, actions, errors)
            bar.numerator = i
            print(bar)

        imageio.mimsave(filename + '.mp4', self.images, fps=5)

    def generate_grid(self, trial, k=1000):
        self.states = np.stack(self.states)
        self.actions = np.stack(self.actions)

        grid, _ = kmeans(whiten(np.concatenate([self.states, self.actions], axis=1)), k)

        np.save('grid_{0}_{1}_{2}'.format(self.config.name, self.config.model, trial), grid)

    def render_frame(self, i, reducer, grid_embedding, states, actions, error):
        figure = plt.figure(figsize=(5.12, 5.12))
        figure.suptitle('Episode ' + str(i))

        plt.subplot(1, 1, 1)
        plt.scatter(grid_embedding[:, 0], grid_embedding[:, 1], marker='o', c=error, cmap='coolwarm', s=8)
        m = plt.cm.ScalarMappable(cmap='coolwarm')
        m.set_array(error)
        m.set_clim(0, self.error_max)
        plt.colorbar(m, boundaries=np.linspace(0, self.error_max, 20))

        trajectory = torch.cat([states, actions], dim=1)
        trajectory_embedding = reducer.transform(trajectory.numpy())
        plt.scatter(trajectory_embedding[:, 0], trajectory_embedding[:, 1], marker='x', c='black', s=8)

        figure.canvas.draw()
        image = np.frombuffer(figure.canvas.tostring_rgb(), dtype='uint8')
        self.images.append(image.reshape(figure.canvas.get_width_height()[::-1] + (3,)))
        plt.close()
