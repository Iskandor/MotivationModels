import os

from pathlib import Path

from agents.PPOAtariAgent import PPOAtariCNDAgent, PPOAtariRNDAgent, PPOAtariFEDRefAgent, PPOAtariICMAgent
from analytic.FeatureAnalysis import FeatureAnalysis
from analytic.MetricTensor import initialize
from analytic.StateCollector import collect_states, collect_samples, save_states
from plots.paths import models_root, states_root, data_root

if __name__ == '__main__':
    results_path = Path(data_root) / 'ppo'

    # states = []
    # next_states = []
    #
    # for i in range(8):
    #     agent, env, _ = initialize(os.path.join(models_root, 'montezuma_42_cnd_{0:d}'.format(i)), '42', PPOAtariCNDAgent)
    #     s0, s1 = collect_states(agent, env, 1000)
    #     states.append(s0)
    #     next_states.append(s1)
    #
    # save_states(states_root, states, next_states)

    config = []

    # agent, _, _ = initialize(os.path.join(models_root, 'montezuma_23_fed_ref_0'), '23', PPOAtariFEDRefAgent)
    # collect_samples(agent, Path(states_root) / 'states.npy', Path(states_root) / 'fed_ref23_0', 'fed_ref')

    # agent, _, _ = initialize(os.path.join(models_root, 'montezuma_2_rnd_0'), '2', PPOAtariRNDAgent)
    # collect_samples(agent, Path(states_root) / 'states.npy', Path(states_root) / 'rnd2_0', 'rnd')

    # agent, _, _ = initialize(os.path.join(models_root, 'montezuma_42_cnd_0'), '42', PPOAtariCNDAgent)
    # collect_samples(agent, Path(states_root) / 'states.npy', Path(states_root) / 'cnd42_0', 'cnd')

    # agent, _, _ = initialize(os.path.join(models_root, 'montezuma_30_icm_0'), '30', PPOAtariICMAgent)
    # collect_samples(agent, Path(states_root) / 'states.npy', Path(states_root) / 'icm30_0', 'icm')

    config_id = 40
    for i in range(8):
        # agent, _, _ = initialize(os.path.join(models_root, 'montezuma_{0:d}_cnd_{1:d}'.format(config_id, i)), '{0:d}'.format(config_id), PPOAtariCNDAgent)
        # collect_samples(agent, Path(states_root) / 'states.npy', Path(states_root) / 'cnd{0:d}_{1:d}'.format(config_id, i), 'cnd')
        config.append({'samples': Path(states_root) / 'cnd{0:d}_{1:d}.npy'.format(config_id, i), 'results': os.path.join(results_path, 'cnd/montezuma/{0:d}/ppo_montezuma_{1:d}_cnd_{2:d}.npy'.format(config_id, config_id, i)),
                       'label': 'cnd{0:d}_{1:d}'.format(config_id, i)})

    # config = [
    #     {'samples': Path(states_root) / 'fed_ref23_0.npy', 'results': os.path.join(results_path, 'fed_ref/montezuma/23/ppo_montezuma_23_fed_ref_0.npy'), 'label': 'fed_ref'},
    #     {'samples': Path(states_root) / 'rnd2_0.npy', 'results': os.path.join(results_path, 'rnd/montezuma/2/ppo_montezuma_2_rnd_0.npy'), 'label': 'rnd'},
    #     {'samples': Path(states_root) / 'cnd42_0.npy', 'results': os.path.join(results_path, 'cnd/montezuma/42/ppo_montezuma_42_cnd_0.npy'), 'label': 'cnd'},
    #     {'samples': Path(states_root) / 'icm30_0.npy', 'results': os.path.join(results_path, 'icm/montezuma/30/ppo_montezuma_30_icm_0.npy'), 'label': 'icm'},
    # ]

    analysis = FeatureAnalysis(config)
    analysis.plot('features_cnd40')
    # analysis.table(filename='features_cnd{0:d}_table'.format(config_id))
    # analysis.plot_feature_boxplot('features_cnd{0:d}_boxplot'.format(config_id))

    # pdf = FPDF()
    # # imagelist is the list with all image filenames
    # for image in ['features_cnd{0:d}_chart.png'.format(config_id), 'features_cnd{0:d}_table.png'.format(config_id), 'features_cnd{0:d}_boxplot.png'.format(config_id)]:
    #     pdf.add_page()
    #     pdf.image(image)
    # pdf.output('features_cnd{0:d}.pdf'.format(config_id), "F")
