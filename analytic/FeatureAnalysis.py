from math import sqrt

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm


class FeatureAnalysis:
    def __init__(self, config):
        self.data = []
        self.results = []
        self.labels = []
        self.colors = ['blue', 'red', 'green', 'magenta', 'cyan', 'orange', 'purple', 'gray', 'navy', 'maroon', 'brown', 'apricot', 'olive', 'beige', 'yellow']
        for row in config:
            self.data.append(np.load(row['samples'], allow_pickle=True).item())
            self.results.append(np.load(row['results'], allow_pickle=True).item())
            self.labels.append(row['label'])

    def table(self, axes=None, filename=''):
        row_labels = self.labels
        col_labels = ['max reward', 'mean reward', 'spd', 'ev 25%', 'ev 50%', 'ev 75%', 'ev 95%']

        cell_text = []

        data = {'re_max': [], 're_mean': [], 'spd': [], 'ev p25': [], 'ev p50': [], 'ev p75': [], 'ev p95': []}

        for r in self.results:
            data['re_max'].append(r['re']['sum'].max())
            data['re_mean'].append(r['re']['sum'].mean())

        for d in self.data:
            data['spd'].append(d['diff'].std())
            pca = PCA()
            pca.fit(d['feature'])
            eigenvalues = pca.singular_values_ ** 2
            p = np.percentile(eigenvalues, [25, 50, 75, 95])
            data['ev p25'].append(p[0])
            data['ev p50'].append(p[1])
            data['ev p75'].append(p[2])
            data['ev p95'].append(p[3])

        sort_index = np.argsort(np.array(data['re_max'])).tolist()
        row_labels = np.array(row_labels)[sort_index]
        for k in data.keys():
            data[k] = np.array(data[k])[sort_index]

        for i in range(len(row_labels)):
            cell_text.append([])
            for j, k in enumerate(data.keys()):
                cell_text[i].append('{0:.4f}'.format(data[k][i]))

        # plt.show()
        if filename != '':
            plt.figure(figsize=(10.24, 5.12))
            plt.tight_layout()
            plt.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc='center')
            plt.axis('off')
            plt.savefig(filename)
        else:
            axes.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc='center')
            axes.axis('off')

    def plot(self, filename):
        data = {'max': [], 'diff_mean': [], 'diff_std': [], 'feature': []}
        for r in self.results:
            data['max'].append(r['re']['sum'].max())

        for d in self.data:
            data['diff_mean'].append(d['diff'].mean())
            data['diff_std'].append(d['diff'].std())
            data['feature'].append(d['feature'])

        fig = plt.figure(figsize=(3 * 7.68, 2 * 7.68))
        # plt.tight_layout()

        ax = fig.add_subplot(311)
        self.table(ax)
        fig.add_subplot(323)
        self.mean_std(data['diff_mean'], data['diff_std'])
        fig.add_subplot(324)
        self.pca(data['feature'])
        fig.add_subplot(325)
        self.max_reward(data['max'])
        fig.add_subplot(326)
        self.ev_boxplot(data['feature'])
        plt.savefig('./{0:s}.png'.format(filename))

    def plot_feature_boxplot(self, filename):
        fig = plt.figure(figsize=(40.96, len(self.data) * 2.56))
        plt.tight_layout()

        flierprops = dict(marker='o', markerfacecolor='green', markersize=2, linestyle='none')

        for i in tqdm(range(len(self.data)), total=len(self.data)):
            f = self.data[i]['feature']
            dim = f.shape[1]
            pca = PCA()
            pca.fit(f)
            y = np.squeeze(pca.transform(np.ones((1, dim))))
            b = np.flip(np.abs(y).argsort())
            f = f[:, b]

            plt.subplot2grid(shape=(len(self.data), 1), loc=(i, 0))
            plt.boxplot(f, flierprops=flierprops, showfliers=False)
        plt.savefig('./{0:s}.png'.format(filename))

    def mean_std(self, diff_mean, diff_std):
        plt.gca().set_title('Std. deviation of feature difference')
        # plt.yscale('log')
        plt.ylabel('feature difference sigma')
        plt.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False)
        max_value = max(diff_std)
        plt.ylim(0, max_value * 2)
        for i, d in enumerate(self.data):
            # plt.errorbar(i, diff_mean[i], yerr=diff_std[i], linestyle='None', marker='o', label=self.labels[i], color=self.colors[i])
            plt.bar(i * 0.5, height=diff_std[i], width=0.3, label=self.labels[i], color=self.colors[i])
            plt.grid()
        plt.legend(ncol=int(sqrt(len(diff_std))))


    def pca(self, features):
        plt.gca().set_title('Eigenvalues')
        plt.xlabel('eigenvalue index')
        plt.ylabel('value')
        plt.yscale('log')
        plt.grid()

        for i, f in enumerate(features):
            U, S, V = torch.pca_lowrank(torch.tensor(f), q=512)
            x_axis = np.arange(len(S))

            plt.plot(x_axis, (S ** 2).numpy(), color=self.colors[i], label=self.labels[i], markersize=2)

        # plt.legend()

    def ev_boxplot(self, features):
        # plt.gca().set_title('Eigenvalues boxplot')

        eigenvalues = []

        for i, f in enumerate(features):
            U, S, V = torch.pca_lowrank(torch.tensor(f), q=512)

            eigenvalues.append(S.numpy() ** 2)

        bplot = plt.boxplot(eigenvalues, showfliers=False, labels=self.labels, patch_artist=True)

        for patch, color in zip(bplot['boxes'], self.colors):
            patch.set_facecolor(color)

    def max_reward(self, rewards):
        plt.gca().set_title('Maximal external reward')
        plt.ylabel('value')
        plt.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False)
        for i, r in enumerate(rewards):
            plt.bar(i * 0.5, height=r, width=0.3, color=self.colors[i], label=self.labels[i])
        # plt.legend()

    def dim_boxplot(self):
        plt.gca().set_title('Feature dimension statistics')
        plt.boxplot(self.data)
