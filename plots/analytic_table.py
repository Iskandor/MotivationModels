import numpy as np

from plots import prepare_data
from plots.key_values import key_values


def compute_table_values(config, keys, end=0):
    data = prepare_data(config)

    latex_text = ''

    for i, row in enumerate(data):
        print(config[i]['model'], config[i]['id'])
        for key in keys:
            means = []
            percentile95 = []
            percentile100 = []

            for inst in row:
                if end > 0:
                    for si, s in enumerate(inst[key]['step']):
                        if s > end:
                            break
                    means.append(np.mean(inst[key][key_values[key]][:si], axis=0).item())
                    percentile95.append(np.percentile(inst[key][key_values[key]][:si], 95, axis=0).item())
                    percentile100.append(np.max(inst[key][key_values[key]][:si], axis=0).item())
                else:
                    means.append(np.mean(inst[key][key_values[key]], axis=0).item())
                    percentile95.append(np.percentile(inst[key][key_values[key]], 95, axis=0).item())
                    percentile100.append(np.max(inst[key][key_values[key]], axis=0).item())

            std = np.std(np.array(means))
            mean = np.mean(np.array(means))
            percentile95 = np.mean(np.array(percentile95))
            percentile100 = np.mean(np.array(percentile100))
            # print(key, mean, std, percentile95, percentile100)
            if key == 're':
                latex_text += '& {0:.2f} $\pm$ {1:.2f} '.format(mean, std)  # re
            if key == 'score':
                latex_text += '& {0:.0f} '.format(percentile100)  # score
    print(latex_text)
