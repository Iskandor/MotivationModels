import argparse
import copy
import time
from concurrent.futures.thread import ThreadPoolExecutor

import imageio
import numpy as np
import psutil
import ray
import torch
import umap
from etaprogress.progress import ProgressBar
import matplotlib.pyplot as plt


class VideoRenderer:
    def __init__(self, data=None):
        if data is not None:
            self.grid = data['grid_embedding']
            self.trajectory = data['trajectories']
            self.error_max = data['error_max']
            self.images = [None] * len(self.trajectory)
        else:
            self.grid = None
            self.trajectory = None
            self.error_max = None
            self.images = None
        self.trajectory_embedding = []

    def render_video(self, filename):
        print('Rendering video {0:s}.mp4'.format(filename))
        print('Computing grid embedding')
        reducer = umap.UMAP(n_jobs=psutil.cpu_count(logical=True))
        grid_embedding = reducer.fit_transform(self.grid)

        print('Computing trajectory embedding')
        bar = ProgressBar(len(self.trajectory), max_width=40)
        for i, t in enumerate(self.trajectory):
            states, actions, errors = t
            self.compute_trajectory_embedding(reducer, states, actions)
            bar.numerator = i
            print(bar)

        print('Rendering frames')
        bar = ProgressBar(len(self.trajectory), max_width=40)
        for i, t in enumerate(self.trajectory):
            states, actions, errors = t
            self.render_frame(i, grid_embedding, self.trajectory_embedding, errors)
            bar.numerator = i
            print(bar)

        print('Saving file {0}'.format(filename + '.mp4'))
        imageio.mimsave(filename + '.mp4', self.images, fps=5)

    def compute_trajectory_embedding(self, reducer, states, actions):
        trajectory = np.concatenate([states, actions], axis=1)
        self.trajectory_embedding.append(reducer.transform(trajectory))

    def render_frame(self, i, grid_embedding, trajectory_embedding, error):

        figure = plt.figure(figsize=(5.12, 5.12))
        figure.suptitle('Episode ' + str(i))

        plt.subplot(1, 1, 1)
        plt.scatter(grid_embedding[:, 0], grid_embedding[:, 1], marker='o', c=error, cmap='coolwarm', s=8)
        m = plt.cm.ScalarMappable(cmap='coolwarm')
        m.set_array(error)
        m.set_clim(0, self.error_max)
        plt.colorbar(m, boundaries=np.linspace(0, self.error_max, 20))

        plt.scatter(trajectory_embedding[i][:, 0], trajectory_embedding[i][:, 1], marker='x', c='black', s=8)

        figure.canvas.draw()
        image = np.frombuffer(figure.canvas.tostring_rgb(), dtype='uint8')
        self.images[i] = image.reshape(figure.canvas.get_width_height()[::-1] + (3,))
        plt.show()
        plt.close()

    def render_error(self, index, grid_embedding, error):
        figure = plt.figure(figsize=(5.12, 5.12))
        figure.suptitle('QRND Error')

        plt.subplot(1, 1, 1)
        plt.scatter(grid_embedding[:, 0], grid_embedding[:, 1], marker='o', c=error, cmap='coolwarm', s=8)
        m = plt.cm.ScalarMappable(cmap='coolwarm')
        m.set_array(error)
        plt.colorbar(m)

        plt.savefig('initialization_test_{0:d}.png'.format(index))
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Motivation models learning platform.')
    parser.add_argument('--file', type=str, help='path to data file .npy')
    parser.add_argument('--video-name', type=str, help='path to data file .npy')
    args = parser.parse_args()

    data = np.load(args.file, allow_pickle=True).item()
    video_renderer = VideoRenderer(data)
    video_renderer.render_video(args.video_name)
