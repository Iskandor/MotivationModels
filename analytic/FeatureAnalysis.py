from math import sqrt
from pathlib import Path

import matplotlib
import numpy as np
import torch
import umap
import umap.plot
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

from plots.dataloader import parse_text_file, parse_analytic_file


class FeatureAnalysis:
    def __init__(self, config):
        self.data = []
        self.results = []
        self.labels = []
        self.colors = ['blue', 'red', 'green', 'magenta', 'cyan', 'orange', 'purple', 'gray', 'navy', 'maroon', 'brown', 'olive', 'beige', 'yellow', 'indianred', 'lawngreen', 'skyblue', 'gold']
        for row in config:
            self.data.append(np.load(row['samples'], allow_pickle=True).item())
            path = Path(row['results'])
            if path.suffix.__str__() == '.npy':
                self.results.append(parse_analytic_file(row['results']))
            elif path.suffix.__str__() == '.log':
                self.results.append(parse_text_file(row['results']))

            self.labels.append(row['label'])

    def table(self, axes=None, filename=''):
        row_labels = self.labels
        col_labels = ['max reward', 'mean reward', 'spd mean', 'spd std', 'ev 25%', 'ev 50%', 'ev 75%', 'ev 95%']

        cell_text = []

        data = {'re_max': [], 're_mean': [], 'spd_mean': [], 'spd_std': [], 'ev p25': [], 'ev p50': [], 'ev p75': [], 'ev p95': []}

        for r in self.results:
            if 're' in r:
                key = 're'
            else:
                key = 'ext_reward'

            data['re_max'].append(r[key]['sum'].max())
            data['re_mean'].append(r[key]['sum'].mean())

        for d in self.data:
            data['spd_mean'].append(d['dist'].mean())
            data['spd_std'].append(d['dist'].std())
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
                if k == 'spd_mean' or k == 'spd_std':
                    cell_text[i].append('{0:.2f}'.format(data[k][i]))
                else:
                    cell_text[i].append('{0:.0f}'.format(data[k][i]))
            print('& \\multicolumn{{1}}{{l|}}{{{0:s}}} & \\multicolumn{{1}}{{c}}{{{1:s}}} & \\multicolumn{{1}}{{c}}{{{2:s} $\pm$ {3:s}}} & \\multicolumn{{1}}{{c}}{{{4:s}}} & \\multicolumn{{1}}{{c}}{{{5:s}}} & \\multicolumn{{1}}{{c}}{{{6:s}}} & \\multicolumn{{1}}{{c}}{{{7:s}}} \\\\'.format(
                row_labels[i], cell_text[i][0], cell_text[i][2], cell_text[i][3], cell_text[i][4], cell_text[i][5], cell_text[i][6], cell_text[i][7]))

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
            data['diff_mean'].append(d['dist'].mean())
            data['diff_std'].append(d['dist'].std())
            data['feature'].append(d['feature'])

        # fig = plt.figure(figsize=(3 * 7.68, 2 * 7.68))
        fig = plt.figure(figsize=(7.68, 7.68))
        # plt.tight_layout()

        # ax = fig.add_subplot(311)
        # self.table(ax)
        # fig.add_subplot(323)
        # self.mean_std(data['diff_mean'], data['diff_std'])
        # fig.add_subplot(324)
        # self.pca(data['feature'])
        self.qr_decomposition(data['feature'])
        # fig.add_subplot(325)
        # self.max_reward(data['max'])
        # fig.add_subplot(326)
        # self.ev_boxplot(data['feature'])
        plt.savefig('{0:s}.png'.format(filename))

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
        plt.gca().set_title('Feature difference')
        # plt.yscale('log')
        plt.ylabel('feature difference sigma')
        plt.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False)
        # max_value = max(diff_std)
        # plt.ylim(max_value * -2, max_value * 2)
        for i, d in enumerate(self.data):
            plt.errorbar(i, diff_mean[i], yerr=diff_std[i], elinewidth=2, capsize=5, marker='o', label=self.labels[i], color=self.colors[i])
            # plt.bar(i * 0.5, height=diff_std[i], width=0.3, label=self.labels[i], color=self.colors[i])
        plt.grid()
        # plt.legend(ncol=int(sqrt(len(diff_std))))

    def pca(self, features):
        # plt.gca().set_title('Eigenvalues')
        # plt.xlabel('eigenvalue index')
        # plt.ylabel('magnitude')
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 16}

        matplotlib.rc('font', **font)

        plt.yscale('log')
        plt.grid()

        for i, f in enumerate(features):
            U, S, V = torch.pca_lowrank(torch.tensor(f), q=512)
            x_axis = np.arange(len(S))

            plt.plot(x_axis, (S ** 2).numpy(), color=self.colors[i], label=self.labels[i], markersize=2)

        # plt.legend(ncol=int(sqrt(len(features))), loc=1)

    def qr_decomposition(self, features):
        diagonals = []
        labels = []
        for i, f1 in enumerate(features):
            q1, _ = np.linalg.qr(np.transpose(f1), mode='complete')
            for j, f2 in enumerate(features):
                q2, _ = np.linalg.qr(np.transpose(f2), mode='complete')
                q = np.transpose(q2) @ q1
                d = np.diagonal(q)
                labels.append((i+1)*(j+1))
                diagonals.append(d)

        diagonals = np.array(diagonals)
        labels = np.array(labels)
        mapper = umap.UMAP().fit(diagonals)
        umap.plot.points(mapper, labels=labels)

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
