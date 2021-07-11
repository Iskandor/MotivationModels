import os

from plots.dataloader import prepare_data
from plots.chart import plot_multiple_models, plot_baseline_details, plot_forward_model_details, plot_dop_model_details

root = 'C:/GIT/Experiments/plots'


def plot(config, plot_overview=True, plot_details=False, window=1000):
    data = prepare_data(config)
    algorithm = config[0]['algorithm']
    env = config[0]['env']
    legend = [key['model'] for key in config]

    if plot_overview:
        path = os.path.join(root, algorithm, env)
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, env)
        plot_multiple_models(
            data,
            legend,
            ['blue', 'red', 'green', 'orchid', 'yellow', 'orange', 'darkcyan', 'brown', 'slategray', 'lime'],
            path,
            window)

    if plot_details:
        for index, key in enumerate(config):
            d = data[index]
            model = key['model']
            id = key['id']

            path = os.path.join(root, algorithm, env, model)
            if not os.path.exists(path):
                os.mkdir(path)
            path = os.path.join(path, id)
            if not os.path.exists(path):
                os.mkdir(path)
            path = os.path.join(path, '{0:s}_{1:s}'.format(env, model))
            if model == 'baseline':
                plot_baseline_details(d, path, window=window)
            if model == 'rnd':
                plot_forward_model_details(d, path, window=window)
            if model == 'qrnd':
                plot_forward_model_details(d, path, window=window)
            if model == 'dop':
                plot_dop_model_details(d, path, window=window)