import os

from plots.analytic_chart import plot_multiple_models, plot_detail
from plots.dataloader import prepare_data
from plots.paths import plot_root


def plot(name, config, plot_overview=True, average_per_step=False, has_score=False, plot_details=[], window=1000):
    data = prepare_data(config)
    algorithm = config[0]['algorithm']
    env = config[0]['env']
    legend = ['{0:s} {1:s}'.format(key['model'], key['id']) for key in config]

    if plot_overview:
        path = os.path.join(plot_root, algorithm, env)
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, name)
        plot_multiple_models(
            data,
            legend,
            ['blue', 'red', 'green', 'yellow', 'orange', 'cyan', 'purple', 'gray', 'magenta', 'navy', 'maroon', 'brown', 'apricot', 'olive', 'beige'],
            path,
            window,
            has_score)

        for index, key in enumerate(config):
            if int(key['id']) in plot_details:
                d = data[index]
                model = key['model']
                id = key['id']

                path = os.path.join(plot_root, algorithm, env, model)
                if not os.path.exists(path):
                    os.mkdir(path)
                path = os.path.join(path, id)
                if not os.path.exists(path):
                    os.mkdir(path)
                path = os.path.join(path, '{0:s}_{1:s}'.format(env, model))
                plot_detail(d, path, window)
