import glob
import os

import numpy as np

from plots.paths import data_root


def prepare_data(keys):
    data = []

    for key in keys:
        model = key['model']
        algorithm = key['algorithm']
        env = key['env']
        id = key['id']

        legacy = False
        if 'legacy' in key:
            legacy = key['legacy']

        path = os.path.join(data_root, algorithm, model, env, id)
        # data.append(load_data(path, ['re', 're_raw', 'ri', 'hid', 'aa', 'var', 'error', 'ext_grad', 'reg_grad', 'dop_grad'], ['loss', 'regterm'], ['re', 're_raw', 'ri']))
        if legacy:
            data.append(convert_data(load_data2(path)))
        else:
            data.append(load_analytic_files(path))

    return data


def expand_data_legacy(data):
    d = []
    for r in data:
        list = []
        for a in r:
            list.append(np.full((int(a[0])), a[1]))
        d.append(np.concatenate(list))
    return np.stack(d)


def align_data(data, steps):
    if data.shape[0] < steps:
        zeros = np.zeros(steps - data.shape[0])
        data = np.concatenate([zeros, data])

    return data


def expand_data(data, steps=None):
    d = []
    for i, r in enumerate(data):
        if steps is None:
            d.append(np.full((int(r[0])), r[1]))
        else:
            d.append(np.full((steps[i],) + r.shape, r))
    return np.concatenate(d)


def load_analytic_files(folder):
    print(folder)
    print(glob.glob(str(folder) + '/*.npy'))

    data = []

    for file in glob.glob(str(folder) + '/*.npy'):
        data.append(np.load(file, allow_pickle=True).item())

    return data


def convert_data(data):
    experiments_size = len(data['re'])

    result = []

    for i in range(experiments_size):
        steps = np.expand_dims(np.cumsum(data['steps'][i]), axis=1)
        v = {
            're': {'step': steps, 'sum': np.expand_dims(data['re'][i], axis=1)},
            'score': {'step': steps, 'sum': np.expand_dims(data['score'][i], axis=1)},
            #TODO toto pada pre baseline kedze nema internal reward
            'ri': {'step': steps, 'mean': np.expand_dims(data['ri'][i] / data['steps'][i], axis=1)}
        }
        result.append(v)

    return result


def load_data2(folder):
    print(folder)
    print(glob.glob(str(folder) + '/*.npy'))

    data = None

    for file in glob.glob(str(folder) + '/*.npy'):
        d = np.load(file, allow_pickle=True).item()
        if data is None:
            data = {}
            for k in list(d.keys()):
                data[k] = []

        for k in list(d.keys()):
            data[k].append(d[k])

    return data


def load_data(folder, expand_keys=[], align_keys=[], stack_keys=[]):
    print(folder)
    print(glob.glob(str(folder) + '/*.npy'))

    data = None

    for file in glob.glob(str(folder) + '/*.npy'):
        d = np.load(file, allow_pickle=True).item()
        if data is None:
            data = {}
            for k in list(d.keys()):
                if k != 'steps':
                    data[k] = []

        steps = None
        if 'steps' in d:
            steps = d['steps']
            # del d['steps']

        total_steps = np.sum(steps)

        for k in list(d.keys()):
            if k in expand_keys:
                if k in data and d[k].size > 0:
                    data[k].append(expand_data(d[k], steps))
            elif k in align_keys:
                if k in data and d[k].size > 0:
                    data[k].append(align_data(d[k], total_steps))
            else:
                data[k].append(d[k])

    for k in list(d.keys()):
        if k in stack_keys and len(data[k]) > 0:
            data[k] = np.stack(data[k])

    return data


def load_data_legacy(folder, suffix, expand=False):
    data = {'re': [], 'ri': [], 'fme': [], 'mce': [], 'fmr': [], 'mcr': [], 'vl': [], 'sdm': [], 'ldm': []}

    print(str(folder) + '/*.' + suffix)
    print(glob.glob(str(folder) + '/*.' + suffix))

    for file in glob.glob(str(folder) + '/*.' + suffix):
        if file.find('_re.') != -1:
            data['re'].append(np.load(file))
        if file.find('_ri.') != -1:
            data['ri'].append(np.load(file))
        if file.find('_fme.') != -1:
            data['fme'].append(np.load(file))
        if file.find('_mce.') != -1:
            data['mce'].append(np.load(file))
        if file.find('_fmr.') != -1:
            data['fmr'].append(np.load(file))
        if file.find('_mcr.') != -1:
            data['mcr'].append(np.load(file))
        if file.find('_vl.') != -1:
            data['vl'].append(np.load(file))
        if file.find('_ldm.') != -1:
            data['ldm'].append(np.load(file))
        if file.find('_sdm.') != -1:
            data['sdm'].append(np.load(file))
    result = {}

    if expand:
        if data['re']:
            result['re'] = expand_data_legacy(data['re'])
        if data['ri']:
            result['ri'] = expand_data_legacy(data['ri'])
        if data['vl']:
            result['vl'] = expand_data_legacy(data['vl'])
    else:
        if data['re']:
            result['re'] = np.stack(data['re'])
        if data['ri']:
            result['ri'] = np.stack(data['ri'])

    if data['fme']:
        for i in range(len(data['fme'])): data['fme'][i] = data['fme'][i][:result['re'].shape[1]]
        result['fme'] = np.stack(data['fme'])
    if data['mce']:
        for i in range(len(data['mce'])): data['mce'][i] = data['mce'][i][:result['re'].shape[1]]
        result['mce'] = np.stack(data['mce'])
    if data['fmr']:
        for i in range(len(data['fmr'])): data['fmr'][i] = data['fmr'][i][:result['re'].shape[1]]
        result['fmr'] = np.stack(data['fmr'])
    if data['mcr']:
        for i in range(len(data['mcr'])): data['mcr'][i] = data['mcr'][i][:result['re'].shape[1]]
        result['mcr'] = np.stack(data['mcr'])
    if data['sdm']:
        result['sdm'] = np.stack(data['sdm'])
    if data['ldm']:
        result['ldm'] = np.stack(data['ldm'])

    return result
