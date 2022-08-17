import os

from agents.PPOAtariAgent import PPOAtariCNDAgent, PPOAtariRNDAgent, PPOAtariFEDRefAgent
from analytic.FeatureAnalysis import FeatureAnalysis
from analytic.MetricTensor import initialize
from analytic.StateCollector import collect_states, collect_samples

if __name__ == '__main__':
    models_path = 'C:/Git/Experiments/models'
    results_path = 'C:/Git/Experiments/data/ppo'

    # agent, env, _ = initialize('./models/montezuma_21_cnd_0', '21', PPOAtariCNDAgent)
    # collect_states(agent, env, 10000)

    # agent, _, _ = initialize(os.path.join(models_path, 'montezuma_23_fed_ref_0'), '23', PPOAtariFEDRefAgent)
    # collect_samples(agent, './states.npy', './fed_ref', 'fed_ref')
    # agent, _, _ = initialize(os.path.join(models_path, 'MontezumaRevengeNoFrameskip-v4_rnd_0'), '2', PPOAtariRNDAgent)
    # collect_samples(agent, './states.npy', './rnd', 'rnd')

    config = []

    for i in range(6):
        # agent, _, _ = initialize(os.path.join(models_path, 'montezuma_21_cnd_{0:d}'.format(i)), '26', PPOAtariCNDAgent)
        # collect_samples(agent, './states.npy', './cnd21_{0:d}'.format(i), 'cnd')
        config.append({'samples': './cnd21_{0:d}.npy'.format(i), 'results': os.path.join(results_path, 'cnd/montezuma/21B/ppo_montezuma_21_cnd_{0:d}.npy'.format(i)), 'label': 'cnd21_{0:d}'.format(i)})

    # config = [
    #     {'samples': './fed_ref.npy', 'results': os.path.join(results_path, 'fed_ref/montezuma/23/ppo_montezuma_23_fed_ref_0.npy'), 'label': 'fed_ref'},
    #     {'samples': './rnd.npy', 'results': os.path.join(results_path, 'rnd/montezuma/2/ppo_montezuma_2_rnd_0.npy'), 'label': 'rnd'},
    #     {'samples': './cnd20.npy', 'results': os.path.join(results_path, 'cnd/montezuma/20/ppo_montezuma_20_cnd_0.npy'), 'label': 'cnd20'},
    #     {'samples': './cnd21.npy', 'results': os.path.join(results_path, 'cnd/montezuma/21/ppo_montezuma_21_cnd_0.npy'), 'label': 'cnd21'},
    #     {'samples': './cnd22.npy', 'results': os.path.join(results_path, 'cnd/montezuma/22/ppo_montezuma_22_cnd_0.npy'), 'label': 'cnd22'},
    #     {'samples': './cnd26.npy', 'results': os.path.join(results_path, 'cnd/montezuma/26/ppo_montezuma_26_cnd_0.npy'), 'label': 'cnd26'},
    # ]

    analysis = FeatureAnalysis(config)
    analysis.plot()
