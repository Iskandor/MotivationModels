import glob
import os
from dataclasses import dataclass

from pathlib import Path

import numpy as np

from agents.PPOAtariAgent import PPOAtariCNDAgent, PPOAtariRNDAgent, PPOAtariFEDRefAgent, PPOAtariICMAgent, PPOAtariFWDAgent
from analytic.FeatureAnalysis import FeatureAnalysis
from analytic.MetricTensor import initialize
from analytic.StateCollector import collect_states, collect_samples, save_states
from plots.paths import models_root, states_root, results_path, plot_root


def generate_analytic_data(descriptor):
    print('Generating data for {0:s}'.format(descriptor.env))
    ref_reward = -1
    for i in range(len(descriptor.models)):
        instance_id, max_reward = find_best_agent(descriptor.env, descriptor.models[i], descriptor.config_ids[i])
        descriptor.instance_ids.append(instance_id)

        if ref_reward < max_reward:
            descriptor.ref_model = descriptor.models[i]
            descriptor.ref_agent = descriptor.agents[i]
            descriptor.ref_config_id = descriptor.config_ids[i]
            descriptor.ref_instance_id = instance_id
            ref_reward = max_reward

    generate_ref_states(descriptor.env, descriptor.env_id, descriptor.ref_model, descriptor.ref_agent, descriptor.ref_config_id, [descriptor.ref_instance_id])

    config = []

    for i in range(len(descriptor.models)):
        generate_features(descriptor.env, descriptor.env_id, descriptor.models[i], descriptor.agents[i], descriptor.config_ids[i], descriptor.instance_ids[i])
        config.append(generate_config(descriptor.env, descriptor.models[i], descriptor.config_ids[i], descriptor.instance_ids[i]))

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


def generate_config(env, model, config_id, instance_id):
    config = {'samples': Path(states_root) / '{0:s}_{1:d}_{2:s}_{3:d}.npy'.format(env, config_id, model, instance_id),
              'results': os.path.join(results_path, '{0:s}/{1:s}/{2:d}/ppo_{3:s}_{4:d}_{5:s}_{6:d}.npy'.format(model, env, config_id, env, config_id, model, instance_id)),
              'label': '{0:s}{1:d}_{2:d}'.format(model, config_id, instance_id)}

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
        results.append((instance_id, max_reward))

    results.sort(key=lambda a: a[1])
    return results[-1][0], results[-1][1]


class Descriptor:
    def __init__(self):
        self.env = None
        self.models = []
        self.agents = []
        self.config_ids = []
        self.instance_ids = []
        self.ref_model = None
        self.ref_agent = None
        self.ref_config_id = None
        self.ref_instance_id = None


class MontezumaDescriptor(Descriptor):
    def __init__(self):
        super().__init__()
        self.env = 'montezuma'
        self.env_id = 'MontezumaRevengeNoFrameskip-v4'
        self.models = ['rnd', 'icm', 'cnd', 'fwd']
        self.agents = [PPOAtariRNDAgent, PPOAtariICMAgent, PPOAtariCNDAgent, PPOAtariFWDAgent]
        self.config_ids = [2, 30, 42, 45]


class GravitarDescriptor(Descriptor):
    def __init__(self):
        super().__init__()
        self.env = 'gravitar'
        self.env_id = 'GravitarNoFrameskip-v4'
        self.models = ['rnd', 'icm', 'cnd', 'fwd']
        self.agents = [PPOAtariRNDAgent, PPOAtariICMAgent, PPOAtariCNDAgent, PPOAtariFWDAgent]
        self.config_ids = [2, 10, 11, 12]


class PrivateEyeDescriptor(Descriptor):
    def __init__(self):
        super().__init__()
        self.env = 'private_eye'
        self.env_id = 'PrivateEyeNoFrameskip-v4'
        self.models = ['rnd', 'icm', 'cnd', 'fwd']
        self.agents = [PPOAtariRNDAgent, PPOAtariICMAgent, PPOAtariCNDAgent, PPOAtariFWDAgent]
        self.config_ids = [2, 5, 4, 6]


class PitfallDescriptor(Descriptor):
    def __init__(self):
        super().__init__()
        self.env = 'pitfall'
        self.env_id = 'PitfallNoFrameskip-v4'
        self.models = ['rnd', 'icm', 'cnd', 'fwd']
        self.agents = [PPOAtariRNDAgent, PPOAtariICMAgent, PPOAtariCNDAgent, PPOAtariFWDAgent]
        self.config_ids = [2, 5, 4, 6]


class SolarisDescriptor(Descriptor):
    def __init__(self):
        super().__init__()
        self.env = 'solaris'
        self.env_id = 'SolarisNoFrameskip-v4'
        self.models = ['rnd', 'icm', 'cnd', 'fwd']
        self.agents = [PPOAtariRNDAgent, PPOAtariICMAgent, PPOAtariCNDAgent, PPOAtariFWDAgent]
        self.config_ids = [2, 5, 4, 6]


class VentureDescriptor(Descriptor):
    def __init__(self):
        super().__init__()
        self.env = 'venture'
        self.env_id = 'VentureNoFrameskip-v4'
        self.models = ['rnd', 'icm', 'cnd', 'fwd']
        self.agents = [PPOAtariRNDAgent, PPOAtariICMAgent, PPOAtariCNDAgent, PPOAtariFWDAgent]
        self.config_ids = [2, 6, 4, 7]


if __name__ == '__main__':

    descriptors = [MontezumaDescriptor, GravitarDescriptor, PrivateEyeDescriptor, PitfallDescriptor, SolarisDescriptor, VentureDescriptor]

    for desc in descriptors:
        desc_instance = desc()
        config = generate_analytic_data(desc_instance)
        analysis = FeatureAnalysis(config)
        analysis.plot(str(Path(plot_root) / desc_instance.env))

    # analysis.table(filename='features_cnd{0:d}_table'.format(config_id))
    # analysis.plot_feature_boxplot('features_cnd{0:d}_boxplot'.format(config_id))
