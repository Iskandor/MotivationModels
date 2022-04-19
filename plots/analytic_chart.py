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


def plot_curve(axis, stats, independent_values, color='blue', alpha=1.0, start=0.0, stop=1.0):
    start = int(len(independent_values) * start)
    stop = int(len(independent_values) * stop)
    if 'val' in stats:
        line, = axis.plot(independent_values[start:stop], stats['val'][start:stop], lw=1, color=color, alpha=alpha)

    if 'sum' in stats:
        line, = axis.plot(independent_values[start:stop], stats['sum'][start:stop], lw=1, color=color, alpha=alpha)

    if 'mean' in stats:
        line, = axis.plot(independent_values[start:stop], stats['mean'][start:stop], lw=1, color=color, alpha=alpha)
        if 'std' in stats:
            axis.fill_between(independent_values[start:stop], stats['mean'][start:stop] + stats['std'][start:stop], stats['mean'][start:stop] - stats['std'][start:stop], facecolor=color, alpha=0.3)

    if 'max' in stats:
        axis.plot(independent_values[start:stop], stats['max'][start:stop], lw=2, color=color, alpha=alpha)

    return line


def get_rows_cols(data):
    n = len(data)

    rows = int(sqrt(n))
    cols = n // rows

    return rows, cols


def plot_chart(num_rows, num_cols, index, key, data, window, color, legend, legend_loc=4):
    ax = plt.subplot(num_rows, num_cols, index)
    ax.set_xlabel('steps')
    ax.set_ylabel(legend)
    ax.grid()

    stats = {}
    iv = None

    for k in data[key]:
        if k != 'step':
            iv, stats[k] = prepare_data_instance(data[key]['step'].squeeze(), data[key][k].squeeze(), window)

    # plot_curve(ax, stats, iv, color=color, alpha=1.0, start=0.19, stop=0.22)
    plot_curve(ax, stats, iv, color=color, alpha=1.0, start=0.0, stop=1.0)
    plt.legend([legend], loc=legend_loc)


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

    lines = []

    for index, d in enumerate(data):
        iv, mu, sigma = prepare_data(d, 're', 'sum', window)
        lines.append(plot_curve(ax, {'mean': mu, 'std': sigma}, iv, color=colors[index]))

    ax.legend(lines, legend[:len(data)], loc=4)

    if has_score:
        lines = []

        ax = plt.subplot(num_rows, num_cols, 2)
        ax.set_xlabel('steps')
        ax.set_ylabel('score')
        ax.grid()

        for index, d in enumerate(data):
            iv, mu, sigma = prepare_data(d, 'score', 'sum', window)
            lines.append(plot_curve(ax, {'mean': mu, 'std': sigma}, iv, color=colors[index]))

        ax.legend(lines, legend[:len(data)], loc=4)

    plt.savefig(path + ".png")
    plt.close()


def plot_detail_cnd(data, path, window=1000):
    num_rows, num_cols = get_rows_cols(data[0])

    for i in tqdm(range(len(data))):
        fig = plt.figure(figsize=(num_cols * 7.00, num_rows * 7.00))

        plot_chart(num_rows, num_cols, 1, 're', data[i], window, color='blue', legend='extrinsic reward')
        plot_chart(num_rows, num_cols, 2, 'score', data[i], window, color='blue', legend='score')
        plot_chart(num_rows, num_cols, 3, 'ri', data[i], window, color='red', legend='intrinsic reward')
        plot_chart(num_rows, num_cols, 4, 'error', data[i], window, color='green', legend='error')
        plot_chart(num_rows, num_cols, 5, 'loss_prediction', data[i], window, color='magenta', legend='loss prediction', legend_loc=9)
        plot_chart(num_rows, num_cols, 6, 'loss_target', data[i], window, color='magenta', legend='loss target', legend_loc=9)
        plot_chart(num_rows, num_cols, 7, 'feature_space', data[i], window, color='maroon', legend='feature space')
        plot_chart(num_rows, num_cols, 8, 'ext_value', data[i], window, color='blue', legend='extrinsic value')
        plot_chart(num_rows, num_cols, 9, 'int_value', data[i], window, color='red', legend='intrinsic value')

        plt.savefig("{0:s}_{1:d}.png".format(path, i))
        plt.close()
