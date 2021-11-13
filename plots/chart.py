import matplotlib
import numpy as np
import matplotlib.pyplot as plt
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


def prepare_data(data, window):
    if len(data.shape) == 1:
        sigma = np.zeros_like(data)
        mu = data
    else:
        sigma = data.std(axis=0)
        mu = data.mean(axis=0)
        if window > 1:
            sigma = moving_average(np.concatenate((np.zeros(window - 1), sigma)), window)

    if window > 1:
        mu = moving_average(np.concatenate((np.zeros(window - 1), mu)), window)

    return mu, sigma


def plot_curve(axis, mu, sigma, independent_values, color, alpha=1.0):
    axis.plot(independent_values, mu, lw=1, color=color, alpha=alpha)
    if sigma is not None:
        axis.fill_between(independent_values, mu + sigma, mu - sigma, facecolor=color, alpha=0.1)


def plot_multiple_models(data, legend, colors, path, window=1):
    plt.figure(figsize=(6.40, 5.12))
    plt.xlabel('steps')
    plt.ylabel('external reward')
    plt.grid()

    t = range(data[0]['re'].shape[1])
    for index, d in enumerate(data):
        mu, sigma = prepare_data(d['re'], window)
        plot_curve(plt, mu, sigma, t, colors[index])

    plt.legend(legend[:len(data)], loc=4)

    plt.savefig(path + ".png")
    plt.close()


def plot_baseline(data, path, window=1000):
    color = 'blue'

    plt.figure(figsize=(8.00, 5.12))
    plt.xlabel('steps')
    plt.ylabel('external reward')
    ax = plt.subplot(1, 1, 1)
    ax.grid()

    t = range(data['re'].shape[1])
    mu, sigma = prepare_data(data['re'], window)
    plot_curve(ax, mu, sigma, t, color)

    plt.savefig(path + ".png")
    plt.close()


def plot_baseline_details(data, path, window=1000):
    num_rows = 1
    num_cols = 3
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
        ax.set_xlabel('steps')
        ax.set_ylabel('mean var / epoch')
        ax.grid()

        color_cycle = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        t = range(data['var'].shape[1])
        for j in range(data['var'].shape[2]):
            mu, sigma = prepare_data(data['var'][i, :, j], window)
            plot_curve(ax, mu, sigma, t, color_cycle[j], alpha=0.5)

        ax = plt.subplot(num_rows, num_cols, 3)
        ax.set_xlabel('steps')

        ax.plot(data['diff_mean'][i], color='blue', alpha=0.5)
        ax.plot(data['diff_max'][i], color='red', alpha=0.5)

        plt.savefig("{0:s}_{1:d}.png".format(path, i))
        plt.close()


def plot_forward_model(data, path, window=1000):
    plt.figure(figsize=(8.00, 5.12))
    plt.xlabel('steps')
    plt.ylabel('reward')
    ax = plt.subplot(1, 1, 1)
    ax.grid()

    t = range(data['re'].shape[1])

    mu, sigma = prepare_data(data['re'], window)
    plot_curve(ax, mu, sigma, t, 'blue')
    mu, sigma = prepare_data(data['ri'], window)
    plot_curve(ax, mu, sigma, t, 'red')
    plt.legend(['external reward', 'internal reward'], loc=4)

    plt.savefig(path + ".png")
    plt.close()


def plot_forward_model_details(data, path, window=1000):
    num_rows = 3
    num_cols = 2
    for i in tqdm(range(data['re'].shape[0])):
        fig = plt.figure(figsize=(num_cols * 7.00, num_rows * 7.00))
        ax = plt.subplot(num_rows, num_cols, 1)
        ax.set_xlabel('steps')
        ax.set_ylabel('reward')
        ax.grid()

        t = range(data['re'].shape[1])

        mu, sigma = prepare_data(data['re'][i], window)
        plot_curve(ax, mu, sigma, t, 'blue')
        mu, sigma = prepare_data(data['ri'][i], window)
        plot_curve(ax, mu, sigma, t, 'red')
        plt.legend(['external reward', 'internal reward'], loc=4)

        ax = plt.subplot(num_rows, num_cols, 3)
        ax.set_xlabel('steps')
        ax.set_ylabel('error')
        ax.set_yscale('log', nonpositive='clip')
        ax.grid()

        t = range(data['fme'][i].shape[0])

        mu, sigma = prepare_data(data['fme'][i], window)
        plot_curve(ax, mu, sigma, t, 'green')
        plt.legend(['prediction error'], loc=1)

        ax = plt.subplot(num_rows, num_cols, 5)
        ax.set_xlabel('reward magnitude')
        ax.set_ylabel('log count')
        ax.set_yscale('log', nonpositive='clip')
        ax.grid()
        bins = np.linspace(0, 1, 50)
        ax.hist(data['fme'][i], bins, color='darkcyan')
        plt.legend(['prediction error reward'], loc=1)

        if 'sdm' in data.keys() and 'ldm' in data.keys():
            ax = plt.subplot(num_rows, num_cols, 2)
            c = ax.pcolormesh(data['sdm'][i], cmap='Reds')
            fig.colorbar(c, ax=ax)

            ax = plt.subplot(num_rows, num_cols, 4)
            c = ax.pcolormesh(data['ldm'][i], cmap='Blues')
            fig.colorbar(c, ax=ax)

        plt.savefig("{0:s}_{1:d}.png".format(path, i))
        plt.close()


def plot_forward_inverse_model_details(data, path, window=1000):
    bar = ProgressBar(data['re'].shape[0], max_width=40)
    num_rows = 3
    num_cols = 2
    for i in range(data['re'].shape[0]):
        fig = plt.figure(figsize=(num_cols * 7.00, num_rows * 7.00))
        ax = plt.subplot(num_rows, num_cols, 1)
        ax.set_xlabel('steps')
        ax.set_ylabel('reward')
        ax.grid()

        t = range(data['re'].shape[1])

        mu, sigma = prepare_data(data['re'][i], window)
        plot_curve(ax, mu, sigma, t, 'blue')
        mu, sigma = prepare_data(data['ri'][i], window)
        plot_curve(ax, mu, sigma, t, 'red')
        plt.legend(['external reward', 'internal reward'], loc=4)

        ax = plt.subplot(num_rows, num_cols, 3)
        ax.set_xlabel('steps')
        ax.set_ylabel('error')
        ax.set_yscale('log', nonpositive='clip')
        ax.grid()

        t = range(data['fme'].shape[1])
        mu, sigma = prepare_data(data['fme'][i], window)
        plot_curve(ax, mu, sigma, t, 'green')

        t = range(data['ime'].shape[1])
        mu, sigma = prepare_data(data['ime'][i], window)
        plot_curve(ax, mu, sigma, t, 'orchid')
        plt.legend(['prediction model error', 'inverse model error'], loc=1)

        ax = plt.subplot(num_rows, num_cols, 5)
        ax.set_xlabel('reward magnitude')
        ax.set_ylabel('log count')
        ax.set_yscale('log', nonpositive='clip')
        ax.grid()
        bins = np.linspace(0, 1, 50)
        ax.hist(data['fme'][i], bins, color='darkcyan')
        plt.legend(['prediction error reward'], loc=1)

        if 'sdm' in data.keys() and 'ldm' in data.keys():
            ax = plt.subplot(num_rows, num_cols, 2)
            c = ax.pcolormesh(data['sdm'][i], cmap='Reds')
            fig.colorbar(c, ax=ax)

            ax = plt.subplot(num_rows, num_cols, 4)
            c = ax.pcolormesh(data['ldm'][i], cmap='Blues')
            fig.colorbar(c, ax=ax)

        plt.savefig("{0:s}_{1:d}.png".format(path, i))
        plt.close()

        bar.numerator = i + 1
        print(bar)


def plot_vae_forward_model_details(data, path, window=1000):
    num_subplots = 4
    bar = ProgressBar(data['re'].shape[0], max_width=40)
    for i in range(data['re'].shape[0]):
        plt.figure(figsize=(8.00, 4 * 5.12))
        ax = plt.subplot(num_subplots, 1, 1)
        ax.set_xlabel('steps')
        ax.set_ylabel('reward')
        ax.grid()

        t = range(data['re'].shape[1])

        mu, sigma = prepare_data(data['re'][i], window)
        plot_curve(ax, mu, sigma, t, 'blue')
        mu, sigma = prepare_data(data['ri'][i], window)
        plot_curve(ax, mu, sigma, t, 'red')
        plt.legend(['external reward', 'internal reward'], loc=4)

        ax = plt.subplot(num_subplots, 1, 2)
        ax.set_xlabel('steps')
        ax.set_ylabel('error')
        ax.grid()

        t = range(data['fme'].shape[1])

        mu, sigma = prepare_data(data['fme'][i], window)
        plot_curve(ax, mu, sigma, t, 'green')
        plt.legend(['prediction error'], loc=1)

        ax = plt.subplot(num_subplots, 1, 3)
        ax.set_xlabel('steps')
        ax.set_ylabel('loss value')
        ax.grid()

        t = range(data['vl'].shape[1])

        mu, sigma = prepare_data(data['vl'][i], window)
        plot_curve(ax, mu, sigma, t, 'orchid')
        plt.legend(['VAE loss'], loc=1)

        ax = plt.subplot(num_subplots, 1, 4)
        ax.set_xlabel('reward magnitude')
        ax.set_ylabel('log count')
        ax.set_yscale('log', nonpositive='clip')
        ax.grid()
        bins = np.linspace(0, 1, 50)
        ax.hist(data['fme'][i], bins, color='darkcyan')
        plt.legend(['prediction error reward'], loc=1)

        plt.savefig("{0:s}_{1:d}.png".format(path, i))
        plt.close()

        bar.numerator = i + 1
        print(bar)


def plot_surprise_model(data, path, window=1000):
    plot_forward_model(data, path, window)


def plot_gated_model(data, path, window=1000):
    plot_forward_model(data, path, window)


def plot_gated_model_details(data, path, window=1000):
    bar = ProgressBar(data['re'].shape[0], max_width=40)
    num_subplots = 4
    for i in range(data['re'].shape[0]):
        plt.figure(figsize=(5.12, num_subplots * 5.12))
        ax = plt.subplot(num_subplots, 1, 1)
        ax.set_xlabel('steps')
        ax.set_ylabel('reward')
        ax.grid()

        t = range(data['re'].shape[1])

        mu, sigma = prepare_data(data['re'][i], window)
        plot_curve(ax, mu, sigma, t, 'blue')
        mu, sigma = prepare_data(data['ri'][i], window)
        plot_curve(ax, mu, sigma, t, 'red')
        plt.legend(['external reward', 'intrinsic reward'], loc=4)

        ax = plt.subplot(num_subplots, 1, 2)
        ax.set_xlabel('steps')
        ax.set_ylabel('error')
        ax.grid()

        t = range(data['fme'].shape[1])

        mu, sigma = prepare_data(data['fme'][i], window)
        plot_curve(ax, mu, sigma, t, 'green', alpha=0.9)
        mu, sigma = prepare_data(data['mce'][i], window)
        plot_curve(ax, mu, sigma, t, 'orchid', alpha=0.9)
        plt.legend(['prediction error', 'prediction error estimate'], loc=1)

        # ax = plt.subplot(num_subplots, 1, 3)
        # ax.set_xlabel('steps')
        # ax.set_ylabel('log error')
        # ax.set_yscale('log', nonpositive='clip')
        # ax.grid()
        #
        # t = range(data['fme'].shape[1])
        #
        # mu, sigma = prepare_data(data['fme'][i], window)
        # plot_curve(ax, mu, sigma, t, 'green', alpha=0.9)
        # mu, sigma = prepare_data(data['mce'][i], window)
        # plot_curve(ax, mu, sigma, t, 'orchid', alpha=0.9)
        # plt.legend(['prediction error', 'prediction error estimate'], loc=1)

        ax = plt.subplot(num_subplots, 1, 3)
        ax.set_xlabel('steps')
        ax.set_ylabel('reward')
        ax.grid()

        t = range(data['fmr'].shape[1])
        mu, sigma = prepare_data(data['mcr'][i], window)
        plot_curve(ax, mu, sigma, t, 'orange', alpha=0.9)
        mu, sigma = prepare_data(data['fmr'][i], window)
        plot_curve(ax, mu, sigma, t, 'darkcyan', alpha=0.9)
        plt.legend(['predictive surprise reward', 'prediction error reward'], loc=1)

        ax = plt.subplot(num_subplots, 1, 4)
        ax.set_xlabel('reward magnitude')
        ax.set_ylabel('log count')
        ax.set_yscale('log', nonpositive='clip')
        ax.grid()

        bins = np.linspace(0, 1, 50)
        ax.hist([data['mcr'][i], data['fmr'][i]], bins, color=['orange', 'darkcyan'], alpha=0.7)
        plt.legend(['predictive surprise reward', 'prediction error reward'], loc=1)

        plt.savefig("{0:s}_{1:d}.png".format(path, i))
        plt.close()

        bar.numerator = i + 1
        print(bar)


def plot_m2_model_details(data, path, window=1000):
    bar = ProgressBar(data['re'].shape[0], max_width=40)
    num_rows = 2
    num_cols = 2
    for i in range(data['re'].shape[0]):
        fig = plt.figure(figsize=(num_cols * 7.00, num_rows * 7.00))
        ax = plt.subplot(num_rows, num_cols, 1)
        ax.set_xlabel('steps')
        ax.set_ylabel('reward')
        ax.grid()

        t = range(data['re'].shape[1])

        mu, sigma = prepare_data(data['re'][i], window)
        plot_curve(ax, mu, sigma, t, 'blue')
        mu, sigma = prepare_data(data['ri'][i], window)
        plot_curve(ax, mu, sigma, t, 'red')
        plt.legend(['external reward', 'internal reward'], loc=4)

        ax = plt.subplot(num_rows, num_cols, 2)
        ax.set_xlabel('steps')
        ax.set_ylabel('weight')
        ax.grid()

        t = range(data['re'].shape[1])

        mu, sigma = prepare_data(data['m2w'][i][:, 0], window)
        plot_curve(ax, mu, sigma, t, 'blue')
        mu, sigma = prepare_data(data['m2w'][i][:, 1], window)
        plot_curve(ax, mu, sigma, t, 'red')
        plt.legend(['curiosity reward', 'familiarity reward'], loc=4)

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
        ax.set_xlabel('reward magnitude')
        ax.set_ylabel('log count')
        ax.set_yscale('log', nonpositive='clip')
        ax.grid()
        bins = np.linspace(0, 1, 50)
        ax.hist(data['fme'][i], bins, color='darkcyan')
        plt.legend(['prediction error reward'], loc=1)

        plt.savefig("{0:s}_{1:d}.png".format(path, i))
        plt.close()

        bar.numerator = i + 1
        print(bar)


def plot_vdop_model_details(data, path, window=1000):
    num_rows = 3
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
        t = range(data['hid'][i].shape[0])
        data_hid = np.divide(data['hid'][i], hid_norm[i])
        unstacked_data = []
        for j in range(data['hid'][i].shape[1]):
            mu, _ = prepare_data(data_hid[:, j], window)
            unstacked_data.append(mu)

        ax.stackplot(t, np.stack(unstacked_data), colors=color_cycle)
        ax.grid()

        ax = plt.subplot(num_rows, num_cols, 3)
        ax.set_xlabel('steps')
        ax.set_ylabel('error')
        # ax.set_yscale('log', nonpositive='clip')
        ax.grid()

        t = range(data['fme'][i].shape[0])

        for j in range(data['fme'][i].shape[1]):
            mu, sigma = prepare_data(data['fme'][i][:, j], window)
            plot_curve(ax, mu, sigma, t, color_cycle[j])
        plt.legend(['RND error'], loc=1)

        ax = plt.subplot(num_rows, num_cols, 4)
        t = range(data['ext_grad'][i].shape[1])

        for j in range(data['ext_grad'][i].shape[0]):
            mu, sigma = prepare_data(data['ext_grad'][i][j], window)
            plot_curve(ax, mu, sigma, t, color_cycle[j])
        ax.grid()

        ax = plt.subplot(num_rows, num_cols, 5)
        t = range(data['dop_grad'][i].shape[1])

        for j in range(data['dop_grad'][i].shape[0]):
            mu, sigma = prepare_data(data['dop_grad'][i][j], window)
            plot_curve(ax, mu, sigma, t, color_cycle[j])
        ax.grid()

        # ax = plt.subplot(num_rows, num_cols, 6)
        # t = range(data['reg_grad'][i].shape[1])
        #
        # for j in range(data['reg_grad'][i].shape[0]):
        #     mu, sigma = prepare_data(data['reg_grad'][i][j], window)
        #     plot_curve(ax, mu, sigma, t, color_cycle[j])
        # ax.grid()

        colors = []
        for head in data['th'][i]:
            colors.append(color_cycle[int(head)])

        ax = plt.subplot(num_rows, num_cols, 7)
        plt.scatter(data['ts'][i][:, 0], data['ts'][i][:, 1], marker='o', c=colors, s=8)

        ax = plt.subplot(num_rows, num_cols, 8)
        heads = data_hid.shape[1]
        colors_all = color_cycle[:heads] * data['ta'][i].shape[0]
        # plt.scatter(data['taa'][i][:, 0], data['taa'][i][:, 1], marker='o', c=colors_all, s=8, alpha=0.3)
        plt.scatter(data['ta'][i][:, 0], data['ta'][i][:, 1], marker='o', c=colors, s=8)

        plt.savefig("{0:s}_{1:d}.png".format(path, i))
        plt.close()


def plot_dop_model_details(data, path, window=1000):
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

        t = range(data['ext_grad'].shape[1])

        ax = plt.subplot(num_rows, num_cols, 4)
        mu, sigma = prepare_data(data['ext_grad'][i], window)
        plot_curve(ax, mu, sigma, t, 'blue')
        mu, sigma = prepare_data(data['dop_grad'][i], window)
        plot_curve(ax, mu, sigma, t, 'red')
        plt.legend(['ext. gradient', 'int. gradient'], loc=1)

        colors = []
        for head in data['th'][i]:
            colors.append(color_cycle[int(head)])

        ax = plt.subplot(num_rows, num_cols, 5)
        plt.scatter(data['ts'][i][:, 0], data['ts'][i][:, 1], marker='o', c=colors, s=8)

        ax = plt.subplot(num_rows, num_cols, 6)
        plt.scatter(data['ta'][i][:, 0], data['ta'][i][:, 1], marker='o', c=colors, s=8)

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


def plot_dop3_model_details(data, path, window=1000):
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

        t = range(data['loss'].shape[1])

        ax = plt.subplot(num_rows, num_cols, 4)
        mu, sigma = prepare_data(data['loss'][i], window)
        plot_curve(ax, mu, sigma, t, 'blue')
        # mu, sigma = prepare_data(data['regterm'][i], window)
        # plot_curve(ax, mu, sigma, t, 'red')
        plt.legend(['motivation loss', 'regularization term'], loc=1)

        colors = []
        for head in data['th'][i]:
            colors.append(color_cycle[int(head)])

        ax = plt.subplot(num_rows, num_cols, 5)
        plt.scatter(data['ts'][i][:, 0], data['ts'][i][:, 1], marker='o', c=colors, s=8)

        ax = plt.subplot(num_rows, num_cols, 6)
        plt.scatter(data['ta'][i][:, 0], data['ta'][i][:, 1], marker='o', c=colors, s=8)

        plt.savefig("{0:s}_{1:d}.png".format(path, i))
        plt.close()
