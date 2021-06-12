import glob
import os

import numpy as np

root = 'C:/GIT/Experiments/data/'


def prepare_data(keys):
    data = []

    for key in keys:
        model = key['model']
        env = key['env']
        id = key['id']

        path = os.path.join(root, model, env, id)
        data.append(load_data(path, ['re', 'ri']))

    return data

def expand_data_legacy(data):
    d = []
    for r in data:
        list = []
        for a in r:
            list.append(np.full((int(a[0])), a[1]))
        d.append(np.concatenate(list))
    return np.stack(d)


def expand_data(data):
    d = []
    for r in data:
        d.append(np.full((int(r[0])), r[1]))
    return np.concatenate(d)


def load_data(folder, expand_keys=[]):
    print(glob.glob(str(folder) + '/*.npy'))

    data = None

    for file in glob.glob(str(folder) + '/*.npy'):
        d = np.load(file, allow_pickle=True).item()
        if data is None:
            data = {}
            for k in list(d.keys()):
                data[k] = []

        for k in list(d.keys()):
            if k in expand_keys:
                data[k].append(expand_data(d[k]))
            else:
                data[k].append(d[k])

    for k in list(d.keys()):
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