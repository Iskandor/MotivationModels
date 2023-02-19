import glob
import os
from dataclasses import dataclass

from pathlib import Path

import numpy as np

from agents.PPOAtariAgent import PPOAtariCNDAgent, PPOAtariRNDAgent, PPOAtariFEDRefAgent, PPOAtariICMAgent, PPOAtariFWDAgent
from agents.PPOProcgenAgent import PPOProcgenRNDAgent, PPOProcgenCNDAgent
from analytic.FeatureAnalysis import FeatureAnalysis
from analytic.MetricTensor import initialize
from analytic.StateCollector import collect_states, collect_samples, save_states
from plots.dataloader import parse_text_line, load_text_files
from plots.paths import models_root, states_root, results_path, plot_root


def generate_analytic_data(descriptor, gen_ref_states=True, gen_features=True):
    print('Generating data for {0:s}'.format(descriptor.env))
    ref_reward = -1
    for i in range(len(descriptor.models)):
        instance_id, max_reward, file = find_best_agent(descriptor.env, descriptor.models[i], descriptor.config_ids[i])
        descriptor.instance_ids.append(instance_id)
        descriptor.result_files.append(file)

        if ref_reward < max_reward:
            descriptor.ref_model = descriptor.models[i]
            descriptor.ref_agent = descriptor.agents[i]
            descriptor.ref_config_id = descriptor.config_ids[i]
            descriptor.ref_instance_id = instance_id
            ref_reward = max_reward

    if gen_ref_states:
        generate_ref_states(descriptor.env, descriptor.env_id, descriptor.ref_model, descriptor.ref_agent, descriptor.ref_config_id, [descriptor.ref_instance_id])

    if gen_features:
        for i in range(len(descriptor.models)):
            generate_features(descriptor.env, descriptor.env_id, descriptor.models[i], descriptor.agents[i], descriptor.config_ids[i], descriptor.instance_ids[i])

    config = []
    for i in range(len(descriptor.models)):
        config.append(generate_config(descriptor.env, descriptor.models[i], descriptor.labels[i], descriptor.result_files[i], descriptor.config_ids[i], descriptor.instance_ids[i]))

    return config


def generate_ref_states(env_name, env_id, model, agent, config_id, instance_id):
    print('Generating reference states')
    states = []
    next_states = []

    for i in instance_id:
        agent, env, _ = initialize(env_name, env_id, os.path.join(models_root, '{0:s}_{1:d}_{2:s}_{3:d}'.format(env_name, config_id, model, i)), str(config_id), agent)
        s0, s1 = collect_states(agent, env, 10000)
        states.append(s0)
        next_states.append(s1)

    save_states(Path(states_root) / '{0:s}.npy'.format(env_name), states, next_states)


def generate_features(env_name, env_id, model, agent, config_id, instance_id):
    print('Generating features for {0:s}'.format(model))
    agent, _, _ = initialize(env_name, env_id, os.path.join(models_root, '{0:s}_{1:d}_{2:s}_{3:d}'.format(env_name, config_id, model, instance_id)), str(config_id), agent)
    collect_samples(agent, Path(states_root) / '{0:s}.npy'.format(env_name), Path(states_root) / '{0:s}_{1:d}_{2:s}_{3:d}'.format(env_name, config_id, model, instance_id), model)


def generate_config(env, model, label, result_file, config_id, instance_id):
    config = {'samples': Path(states_root) / '{0:s}_{1:d}_{2:s}_{3:d}.npy'.format(env, config_id, model, instance_id),
              # 'results': os.path.join(results_path, '{0:s}/{1:s}/{2:d}/ppo_{3:s}_{4:d}_{5:s}_{6:d}.npy'.format(model, env, config_id, env, config_id, model, instance_id)),
              'results': result_file,
              'id': '{0:s}{1:d}_{2:d}'.format(model, config_id, instance_id),
              'label': label,
              }

    return config


def find_best_agent(env, model, config_id):
    folder = os.path.join(results_path, '{0:s}/{1:s}/{2:d}/'.format(model, env, config_id))
    results = []
    for file in glob.glob(str(folder) + '/*.npy'):
        path = Path(file)
        instance_id = int(path.stem.split('_')[-1])
        data = np.load(file, allow_pickle=True).item()

        if 're' in data:
            key = 're'
        else:
            key = 'ext_reward'

        max_reward = np.max(data[key]['sum']).item()
        results.append((instance_id, max_reward, file))

    data = load_text_files(folder)
    if len(data) > 0:
        for i, file in enumerate(glob.glob(str(folder) + '/*.log')):
            path = Path(file)
            instance_id = int(path.stem.split('_')[-1])
            max_reward = int(np.max(data[i]['re']['sum']).item())
            results.append((instance_id, max_reward, file))

    results.sort(key=lambda a: a[1])
    return results[-1][0], results[-1][1], results[-1][2]


class Descriptor:
    def __init__(self):
        self.env = None
        self.models = []
        self.agents = []
        self.config_ids = []
        self.instance_ids = []
        self.result_files = []
        self.labels = []
        self.ref_model = None
        self.ref_agent = None
        self.ref_config_id = None
        self.ref_instance_id = None


class MontezumaDescriptor(Descriptor):
    def __init__(self):
        super().__init__()
        self.env = 'montezuma'
        self.env_id = 'MontezumaRevengeNoFrameskip-v4'
        self.models = ['rnd', 'cnd', 'cnd', 'cnd']
        self.agents = [PPOAtariRNDAgent, PPOAtariCNDAgent, PPOAtariCNDAgent, PPOAtariCNDAgent]
        self.config_ids = [2, 49, 42, 44]
        self.labels = ['RND', 'SND-V', 'SND-STD', 'SND-VIC']


class GravitarDescriptor(Descriptor):
    def __init__(self):
        super().__init__()
        self.env = 'gravitar'
        self.env_id = 'GravitarNoFrameskip-v4'
        self.models = ['rnd', 'cnd', 'cnd', 'cnd']
        self.agents = [PPOAtariRNDAgent, PPOAtariCNDAgent, PPOAtariCNDAgent, PPOAtariCNDAgent]
        self.config_ids = [2, 14, 11, 13]
        self.labels = ['RND', 'SND-V', 'SND-STD', 'SND-VIC']


class PrivateEyeDescriptor(Descriptor):
    def __init__(self):
        super().__init__()
        self.env = 'private_eye'
        self.env_id = 'PrivateEyeNoFrameskip-v4'
        self.models = ['rnd', 'cnd', 'cnd', 'cnd']
        self.agents = [PPOAtariRNDAgent, PPOAtariCNDAgent, PPOAtariCNDAgent, PPOAtariCNDAgent]
        self.config_ids = [2, 8, 4, 7]
        self.labels = ['RND', 'SND-V', 'SND-STD', 'SND-VIC']


class PitfallDescriptor(Descriptor):
    def __init__(self):
        super().__init__()
        self.env = 'pitfall'
        self.env_id = 'PitfallNoFrameskip-v4'
        self.models = ['rnd', 'cnd', 'cnd', 'cnd']
        self.agents = [PPOAtariRNDAgent, PPOAtariCNDAgent, PPOAtariCNDAgent, PPOAtariCNDAgent]
        self.config_ids = [2, 8, 4, 7]
        self.labels = ['RND', 'SND-V', 'SND-STD', 'SND-VIC']


class SolarisDescriptor(Descriptor):
    def __init__(self):
        super().__init__()
        self.env = 'solaris'
        self.env_id = 'SolarisNoFrameskip-v4'
        self.models = ['rnd', 'cnd', 'cnd', 'cnd']
        self.agents = [PPOAtariRNDAgent, PPOAtariCNDAgent, PPOAtariCNDAgent, PPOAtariCNDAgent]
        self.config_ids = [2, 8, 4, 7]
        self.labels = ['RND', 'SND-V', 'SND-STD', 'SND-VIC']


class VentureDescriptor(Descriptor):
    def __init__(self):
        super().__init__()
        self.env = 'venture'
        self.env_id = 'VentureNoFrameskip-v4'
        self.models = ['rnd', 'cnd', 'cnd', 'cnd']
        self.agents = [PPOAtariRNDAgent, PPOAtariCNDAgent, PPOAtariCNDAgent, PPOAtariCNDAgent]
        self.config_ids = [2, 10, 4, 8]
        self.labels = ['RND', 'SND-V', 'SND-STD', 'SND-VIC']


class CaveflyerDescriptor(Descriptor):
    def __init__(self):
        super().__init__()
        self.env = 'caveflyer'
        self.env_id = 'procgen-caveflyer-v0'
        self.models = ['rnd', 'cnd', 'cnd', 'cnd']
        self.agents = [PPOProcgenRNDAgent, PPOProcgenCNDAgent, PPOProcgenCNDAgent, PPOProcgenCNDAgent]
        self.config_ids = [2, 8, 4, 7]
        self.labels = ['RND', 'SND-V', 'SND-STD', 'SND-VIC']


class ClimberDescriptor(Descriptor):
    def __init__(self):
        super().__init__()
        self.env = 'climber'
        self.env_id = 'procgen-climber-v0'
        self.models = ['rnd', 'cnd', 'cnd', 'cnd']
        self.agents = [PPOProcgenRNDAgent, PPOProcgenCNDAgent, PPOProcgenCNDAgent, PPOProcgenCNDAgent]
        self.config_ids = [2, 8, 4, 7]
        self.labels = ['RND', 'SND-V', 'SND-STD', 'SND-VIC']


class CoinrunDescriptor(Descriptor):
    def __init__(self):
        super().__init__()
        self.env = 'coinrun'
        self.env_id = 'procgen-coinrun-v0'
        self.models = ['rnd', 'cnd', 'cnd', 'cnd']
        self.agents = [PPOProcgenRNDAgent, PPOProcgenCNDAgent, PPOProcgenCNDAgent, PPOProcgenCNDAgent]
        self.config_ids = [2, 8, 4, 7]
        self.labels = ['RND', 'SND-V', 'SND-STD', 'SND-VIC']


class JumperDescriptor(Descriptor):
    def __init__(self):
        super().__init__()
        self.env = 'jumper'
        self.env_id = 'procgen-jumper-v0'
        self.models = ['rnd', 'cnd', 'cnd', 'cnd']
        self.agents = [PPOProcgenRNDAgent, PPOProcgenCNDAgent, PPOProcgenCNDAgent, PPOProcgenCNDAgent]
        self.config_ids = [2, 8, 4, 7]
        self.labels = ['RND', 'SND-V', 'SND-STD', 'SND-VIC']

if __name__ == '__main__':

    # descriptors = [MontezumaDescriptor, GravitarDescriptor, PrivateEyeDescriptor, SolarisDescriptor, VentureDescriptor]
    descriptors = [CaveflyerDescriptor, ClimberDescriptor, CoinrunDescriptor, JumperDescriptor]
    # descriptors = [SolarisDescriptor]

    for desc in descriptors:
        desc_instance = desc()
        config = generate_analytic_data(desc_instance, gen_ref_states=False, gen_features=False)
        analysis = FeatureAnalysis(config)
        analysis.plot(str(Path(plot_root) / desc_instance.env))

    # analysis.table(filename='features_cnd{0:d}_table'.format(config_id))
    # analysis.plot_feature_boxplot('features_cnd{0:d}_boxplot'.format(config_id))
