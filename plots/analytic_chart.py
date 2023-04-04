import math
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
# font = {'family': 'normal',
#         'weight': 'bold',
#         'size': 12}
#
# matplotlib.rc('font', **font)
# matplotlib.rc('axes', titlesize=14)
from tqdm import tqdm

from plots.key_values import key_values

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def prepare_data_instance(data_x, data_y, window, smooth=True):
    dx = data_x

    if len(data_y) > 256:
        dy = np.concatenate((data_y[:-128], data_y[-256:-128]))  # at the end of data is garbage from unfinished 128 environments
    else:
        dy = data_y

    if smooth:
        for i in range(len(dy))[1:]:
            dy[i] = dy[i - 1] * 0.99 + dy[i] * 0.01

    max_steps = int(dx[-1])
    data = np.interp(np.arange(start=0, stop=max_steps, step=window), dx, dy)

    iv = list(range(0, max_steps, window))

    return iv, data


# fwd model bug fix - added synonyms
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
    cols = math.ceil(n / rows)

    return rows, cols


def plot_chart(num_rows, num_cols, index, key, data, value_key, window, color, legend, legend_loc=4):
    ax = plt.subplot(num_rows, num_cols, index)
    ax.set_xlabel('steps')
    ax.set_ylabel(legend)
    ax.grid()

    stats = {}
    iv = None

    for k in value_key:
        iv, stats[k] = prepare_data_instance(data[key]['step'].squeeze(), data[key][k].squeeze(), window)

    # plot_curve(ax, stats, iv, color=color, alpha=1.0, start=0.0, stop=1.0)
    plot_curve(ax, stats, iv, color=color, alpha=1.0, start=0.01, stop=1.0)
    plt.legend([legend], loc=legend_loc)


def plot_multiple_models(keys, data, legend, labels, colors, path, window=1):
    num_rows = 1
    num_cols = len(keys)

    plt.figure(figsize=(num_cols * 6.40, num_rows * 5.12))

    for i, key in enumerate(keys):
        ax = plt.subplot(num_rows, num_cols, i + 1)
        # ax.set_xlabel('steps')
        # ax.set_ylabel(labels[i])
        ax.grid()

        lines = []
        for index, d in enumerate(data):
            iv, mu, sigma = prepare_data(d, key, key_values[key], window)
            iv = [val / 1e6 for val in iv]
            # mu = np.clip(mu, 0, 0.1)
            lines.append(plot_curve(ax, {'mean': mu, 'std': sigma}, iv, color=colors[index], start=0.0))

        if legend is not None:
            ax.legend(lines, legend[:len(data)], loc=0)

    plt.savefig(path + ".png")
    plt.close()


def plot_detail_baseline(data, path, window=1000):
    num_rows, num_cols = get_rows_cols(data[0])

    for i in tqdm(range(len(data))):
        plt.figure(figsize=(num_cols * 7.00, num_rows * 7.00))

        plot_chart(num_rows, num_cols, 1, 're', data[i], ['sum'], window, color='blue', legend='extrinsic reward')
        plot_chart(num_rows, num_cols, 2, 'score', data[i], ['sum'], window, color='blue', legend='score')
        plot_chart(num_rows, num_cols, 3, 'ext_value', data[i], ['mean', 'max', 'std'], window, color='blue', legend='extrinsic value')

        plt.savefig("{0:s}_{1:d}.png".format(path, i))
        plt.close()


def plot_detail_cnd(data, path, window=1000):
    num_rows, num_cols = get_rows_cols(data[0])

    for i in tqdm(range(len(data))):
        plt.figure(figsize=(num_cols * 7.00, num_rows * 7.00))

        plot_chart(num_rows, num_cols, 1, 're', data[i], ['sum'], window, color='blue', legend='extrinsic reward')
        plot_chart(num_rows, num_cols, 2, 'score', data[i], ['sum'], window, color='blue', legend='score')
        plot_chart(num_rows, num_cols, 3, 'ri', data[i], ['mean', 'max', 'std'], window, color='red', legend='intrinsic reward')
        plot_chart(num_rows, num_cols, 4, 'error', data[i], ['mean', 'max', 'std'], window, color='green', legend='error')
        plot_chart(num_rows, num_cols, 5, 'loss_prediction', data[i], ['val'], window, color='magenta', legend='loss prediction', legend_loc=9)
        plot_chart(num_rows, num_cols, 6, 'loss_target', data[i], ['val'], window, color='magenta', legend='loss target', legend_loc=9)
        if 'loss_reg' in data[i]:
            plot_chart(num_rows, num_cols, 7, 'loss_reg', data[i], ['val'], window, color='magenta', legend='loss target reg', legend_loc=9)
        if 'inv_accuracy' in data[i]:
            plot_chart(num_rows, num_cols, 7, 'inv_accuracy', data[i], ['val'], window, color='maroon', legend='inv. model accuracy', legend_loc=9)
        if 'loss_target_norm' in data[i]:
            plot_chart(num_rows, num_cols, 8, 'loss_target_norm', data[i], ['val'], window, color='magenta', legend='loss target norm', legend_loc=9)
        if 'state_space' in data[i]:
            plot_chart(num_rows, num_cols, 9, 'state_space', data[i], ['mean', 'max', 'std'], window, color='maroon', legend='state space', legend_loc=9)
        if 'feature_space' in data[i]:
            plot_chart(num_rows, num_cols, 10, 'feature_space', data[i], ['mean', 'max', 'std'], window, color='maroon', legend='feature space', legend_loc=9)

        plot_chart(num_rows, num_cols, 11, 'ext_value', data[i], ['mean', 'max', 'std'], window, color='blue', legend='extrinsic value')
        plot_chart(num_rows, num_cols, 12, 'int_value', data[i], ['mean', 'max', 'std'], window, color='red', legend='intrinsic value')

        plt.savefig("{0:s}_{1:d}.png".format(path, i))
        plt.close()


def plot_detail_rnd(data, path, window=1000):
    num_rows, num_cols = get_rows_cols(data[0])

    for i in tqdm(range(len(data))):
        plt.figure(figsize=(num_cols * 7.00, num_rows * 7.00))

        plot_chart(num_rows, num_cols, 1, 're', data[i], ['sum'], window, color='blue', legend='extrinsic reward')
        plot_chart(num_rows, num_cols, 2, 'score', data[i], ['sum'], window, color='blue', legend='score')
        plot_chart(num_rows, num_cols, 3, 'ri', data[i], ['mean', 'max', 'std'], window, color='red', legend='intrinsic reward')
        plot_chart(num_rows, num_cols, 4, 'error', data[i], ['mean', 'max', 'std'], window, color='green', legend='error')
        plot_chart(num_rows, num_cols, 5, 'loss_prediction', data[i], ['val'], window, color='magenta', legend='loss prediction', legend_loc=9)
        plot_chart(num_rows, num_cols, 6, 'ext_value', data[i], ['mean', 'max', 'std'], window, color='blue', legend='extrinsic value')
        plot_chart(num_rows, num_cols, 7, 'int_value', data[i], ['mean', 'max', 'std'], window, color='red', legend='intrinsic value')

        plt.savefig("{0:s}_{1:d}.png".format(path, i))
        plt.close()


def plot_detail_icm(data, path, window=1000):
    num_rows, num_cols = get_rows_cols(data[0])

    for i in tqdm(range(len(data))):
        plt.figure(figsize=(num_cols * 7.00, num_rows * 7.00))

        plot_chart(num_rows, num_cols, 1, 're', data[i], ['sum'], window, color='blue', legend='extrinsic reward')
        plot_chart(num_rows, num_cols, 2, 'score', data[i], ['sum'], window, color='blue', legend='score')
        plot_chart(num_rows, num_cols, 3, 'ri', data[i], ['mean', 'max', 'std'], window, color='red', legend='intrinsic reward')
        plot_chart(num_rows, num_cols, 4, 'error', data[i], ['mean', 'max', 'std'], window, color='green', legend='error')
        plot_chart(num_rows, num_cols, 5, 'loss', data[i], ['val'], window, color='magenta', legend='loss', legend_loc=9)
        plot_chart(num_rows, num_cols, 6, 'inverse_loss', data[i], ['val'], window, color='magenta', legend='inverse model loss', legend_loc=9)
        plot_chart(num_rows, num_cols, 7, 'forward_loss', data[i], ['val'], window, color='magenta', legend='forward model loss', legend_loc=9)
        plot_chart(num_rows, num_cols, 8, 'ext_value', data[i], ['mean', 'max', 'std'], window, color='blue', legend='extrinsic value')
        plot_chart(num_rows, num_cols, 9, 'int_value', data[i], ['mean', 'max', 'std'], window, color='red', legend='intrinsic value')
        plot_chart(num_rows, num_cols, 10, 'feature_space', data[i], ['mean', 'max', 'std'], window, color='maroon', legend='feature space', legend_loc=9)

        plt.savefig("{0:s}_{1:d}.png".format(path, i))
        plt.close()


def plot_detail_fwd(data, path, window=1000):
    num_rows, num_cols = get_rows_cols(data[0])

    for i in tqdm(range(len(data))):
        plt.figure(figsize=(num_cols * 7.00, num_rows * 7.00))

        plot_chart(num_rows, num_cols, 1, 're', data[i], ['sum'], window, color='blue', legend='extrinsic reward')
        plot_chart(num_rows, num_cols, 2, 'score', data[i], ['sum'], window, color='blue', legend='score')
        plot_chart(num_rows, num_cols, 3, 'ri', data[i], ['mean', 'max', 'std'], window, color='red', legend='intrinsic reward')
        plot_chart(num_rows, num_cols, 4, 'error', data[i], ['mean', 'max', 'std'], window, color='green', legend='error')
        plot_chart(num_rows, num_cols, 5, 'loss_target', data[i], ['val'], window, color='magenta', legend='loss_target', legend_loc=9)
        plot_chart(num_rows, num_cols, 6, 'loss_target_norm', data[i], ['val'], window, color='magenta', legend='loss_target_norm', legend_loc=9)
        plot_chart(num_rows, num_cols, 7, 'loss_prediction', data[i], ['val'], window, color='magenta', legend='loss_prediction', legend_loc=9)
        plot_chart(num_rows, num_cols, 8, 'ext_value', data[i], ['mean', 'max', 'std'], window, color='blue', legend='extrinsic value')
        plot_chart(num_rows, num_cols, 9, 'int_value', data[i], ['mean', 'max', 'std'], window, color='red', legend='intrinsic value')
        # plot_chart(num_rows, num_cols, 10, 'feature_space', data[i], ['mean', 'max', 'std'], window, color='maroon', legend='feature space', legend_loc=9)

        plt.savefig("{0:s}_{1:d}.png".format(path, i))
        plt.close()
