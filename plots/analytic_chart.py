from math import sqrt

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import umap
from etaprogress.progress import ProgressBar

# font = {'family': 'normal',
#         'weight': 'bold',
#         'size': 12}
#
# matplotlib.rc('font', **font)
# matplotlib.rc('axes', titlesize=14)
from tqdm import tqdm


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def prepare_data_instance(data_x, data_y, window, smooth=True):
    dx = data_x
    dy = data_y

    if smooth:
        for i in range(len(dy))[1:]:
            dy[i] = dy[i - 1] * 0.99 + dy[i] * 0.01

    max_steps = int(dx[-1])
    data = np.interp(np.arange(start=0, stop=max_steps, step=window), dx, dy)

    iv = list(range(0, max_steps, window))

    return iv, data


def prepare_data(data, master_key, key, window):
    steps = []
    values = []
    iv = None

    for d in data:
        steps.append(d[master_key]['step'])
        values.append(d[master_key][key])

    result = []
    for x, y in zip(steps, values):
        iv, value = prepare_data_instance(x.squeeze(), y.squeeze(), window)
        result.append(value)

    result = np.stack(result)
    sigma = result.std(axis=0)
    mu = result.mean(axis=0)

    return iv, mu, sigma


def plot_curve(axis, stats, independent_values, color, alpha=1.0):
    if 'val' in stats:
        axis.plot(independent_values, stats['val'], lw=1, color=color, alpha=alpha)

    if 'sum' in stats:
        axis.plot(independent_values, stats['sum'], lw=1, color=color, alpha=alpha)

    if 'mean' in stats:
        axis.plot(independent_values, stats['mean'], lw=1, color=color, alpha=alpha)
        if 'std' in stats:
            axis.fill_between(independent_values, stats['mean'] + stats['std'], stats['mean'] - stats['std'], facecolor=color, alpha=0.1)

    if 'max' in stats:
        axis.plot(independent_values, stats['max'], lw=2, color=color, alpha=alpha)


def plot_multiple_models(data, legend, colors, path, window=1, has_score=False):
    num_rows = 1
    num_cols = 1

    if has_score:
        num_cols = 2

    plt.figure(figsize=(num_cols * 6.40, num_rows * 5.12))
    ax = plt.subplot(num_rows, num_cols, 1)
    ax.set_xlabel('steps')
    ax.set_ylabel('external reward')
    ax.grid()

    for index, d in enumerate(data):
        iv, mu, sigma = prepare_data(d, 're', 'sum', window)
        plot_curve(ax, {'mean': mu, 'std': sigma}, iv, colors[index])

    ax.legend(legend[:len(data)], loc=4)

    if has_score:
        ax = plt.subplot(num_rows, num_cols, 2)
        ax.set_xlabel('steps')
        ax.set_ylabel('score')
        ax.grid()

        for index, d in enumerate(data):
            iv, mu, sigma = prepare_data(d, 'score', 'sum', window)
            plot_curve(ax, {'mean': mu, 'std': sigma}, iv, colors[index])

        ax.legend(legend[:len(data)], loc=4)

    plt.savefig(path + ".png")
    plt.close()


def get_rows_cols(data):
    n = len(data)

    rows = int(sqrt(n))
    cols = n // rows

    return rows, cols


def plot_chart(num_rows, num_cols, index, key, data, window):
    ax = plt.subplot(num_rows, num_cols, index)
    ax.set_xlabel('steps')
    ax.set_ylabel(key)
    ax.grid()

    stats = {}
    iv = None

    for k in data[key]:
        if k != 'step':
            iv, stats[k] = prepare_data_instance(data[key]['step'].squeeze(), data[key][k].squeeze(), window)

    plot_curve(ax, stats, iv, 'blue')
    plt.legend(['dummy'], loc=4)


def plot_detail(data, path, window=1000):
    num_rows, num_cols = get_rows_cols(data[0])

    for i in tqdm(range(len(data))):
        fig = plt.figure(figsize=(num_cols * 7.00, num_rows * 7.00))
        index = 1

        for k in data[i]:
            plot_chart(num_rows, num_cols, index, k, data[i], window)
            index += 1

        plt.savefig("{0:s}_{1:d}.png".format(path, i))
        plt.close()


def plot_dop2_model_details(data, path, window=1000):
    num_rows = 2
    num_cols = 3

    hid_norm = np.expand_dims(np.sum(data['hid'], axis=2), 2)

    for i in tqdm(range(data['re'].shape[0])):
        fig = plt.figure(figsize=(num_cols * 7.00, num_rows * 7.00))
        ax = plt.subplot(num_rows, num_cols, 1)
        ax.set_xlabel('steps')
        ax.set_ylabel('reward')
        ax.grid()

        t = range(data['re'].shape[1])

        mu, sigma = prepare_data(data['re'][i], window)
        plot_curve(ax, mu, sigma, t, 'blue')
        plt.legend(['external reward'], loc=4)

        ax = plt.subplot(num_rows, num_cols, 2)
        color_cycle = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        t = range(data['hid'].shape[1])
        data_hid = np.divide(data['hid'][i], hid_norm[i])
        unstacked_data = []
        for j in range(data['hid'].shape[2]):
            mu, _ = prepare_data(data_hid[:, j], window)
            unstacked_data.append(mu)

        ax.stackplot(t, np.stack(unstacked_data), colors=color_cycle)
        ax.grid()

        ax = plt.subplot(num_rows, num_cols, 3)
        ax.set_xlabel('steps')
        ax.set_ylabel('error')
        ax.set_yscale('log', nonpositive='clip')
        ax.grid()

        t = range(data['fme'].shape[1])

        mu, sigma = prepare_data(data['fme'][i], window)
        plot_curve(ax, mu, sigma, t, 'green')
        plt.legend(['prediction error'], loc=1)

        ax = plt.subplot(num_rows, num_cols, 4)
        ax.set_xlabel('steps')
        ax.grid()
        t = range(data['aa'].shape[1])
        mu, sigma = prepare_data(data['aa'][i], window)
        plot_curve(ax, mu, sigma, t, 'darkcyan')
        plt.legend(['arbiter accuracy'], loc=1)

        colors = []
        for head in data['th'][i]:
            colors.append(color_cycle[int(head)])

        ax = plt.subplot(num_rows, num_cols, 5)
        plt.scatter(data['ts'][i][:, 0], data['ts'][i][:, 1], marker='o', c=colors, s=8)

        ax = plt.subplot(num_rows, num_cols, 6)
        plt.scatter(data['ta'][i][:, 0], data['ta'][i][:, 1], marker='o', c=colors, s=8)

        plt.savefig("{0:s}_{1:d}.png".format(path, i))
        plt.close()
