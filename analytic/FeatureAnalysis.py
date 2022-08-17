import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


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

    def plot(self):
        data = {'max': [], 'diff_mean': [], 'diff_std': [], 'feature': []}
        for r in self.results:
            data['max'].append(r['re']['sum'].max())

        for d in self.data:
            data['diff_mean'].append(d['diff'].mean())
            data['diff_std'].append(d['diff'].std())
            data['feature'].append(d['feature'])

        fig = plt.figure(figsize=(7.68, 3 * 5.12))
        plt.tight_layout()

        fig.add_subplot(311)
        self.mean_std(data['diff_mean'], data['diff_std'])
        fig.add_subplot(312)
        self.pca(data['feature'])
        fig.add_subplot(313)
        self.max_reward(data['max'])
        plt.savefig('./feature_analysis.png')

    def mean_std(self, diff_mean, diff_std):
        plt.gca().set_title('Std. deviation of feature difference')
        plt.yscale('log')
        plt.ylabel('feature difference sigma')
        plt.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False)
        for i, d in enumerate(self.data):
            # plt.errorbar(i, diff_mean[i], yerr=diff_std[i], linestyle='None', marker='o', label=self.labels[i], color=self.colors[i])
            plt.bar(i * 0.5, height=diff_std[i], width=0.3, label=self.labels[i], color=self.colors[i])
            plt.grid()
            plt.legend()

    def pca(self, features):
        plt.gca().set_title('Singular values')
        plt.xlabel('singular value index')
        plt.ylabel('value')
        # plt.yscale('log')
        plt.grid()

        for i, f in enumerate(features):
            pca = PCA()
            pca.fit(f - f.mean())
            x_axis = np.arange(len(pca.singular_values_))

            plt.plot(x_axis, pca.singular_values_, color=self.colors[i], label=self.labels[i])

        plt.legend()

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
        plt.legend()