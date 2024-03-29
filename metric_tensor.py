from agents.PPOAtariAgent import PPOAtariCNDAgent, PPOAtariRNDAgent
from analytic.MetricTensor import compute_metric_tensor, plot_tensors, MetricTensor, initialize
from analytic.StateCollector import collect_states, collect_samples

if __name__ == '__main__':
    agent, env, _ = initialize('./models/montezuma_21_cnd_0', '21', PPOAtariCNDAgent)
    collect_states(agent, env, 10000)

    agent, _, _ = initialize('./models/montezuma_21_cnd_0', '21', PPOAtariCNDAgent)
    collect_samples(agent, './states.npy', './base', 'baseline')
    agent, _, _ = initialize('./models/MontezumaRevengeNoFrameskip-v4_rnd_0', '2', PPOAtariRNDAgent)
    collect_samples(agent, './states.npy', './rnd', 'rnd')
    agent, _, _ = initialize('./models/montezuma_21_cnd_0', '21', PPOAtariCNDAgent)
    collect_samples(agent, './states.npy', './cnd', 'cnd')

    compute_metric_tensor('./base.npy', './base.npy', './tensor_base', batch=10000, gpu=0, lr=1e-3)
    compute_metric_tensor('./rnd.npy', './rnd.npy', './tensor_rnd', batch=10000, gpu=0, lr=1e-3)
    compute_metric_tensor('./cnd.npy', './cnd.npy', './tensor_cnd', batch=10000, gpu=0, lr=1e-3)

    # plot_tensors(['./tensor_base.npy', './tensor_rnd.npy', './tensor_cnd.npy'])

    # metric_tensor = MetricTensor('cuda:0')
    # mt = metric_tensor.load('./tensor_base.npy')
    # vol = metric_tensor.volume(mt)
    # print(vol)
    # mt = metric_tensor.load('./tensor_rnd.npy')
    # vol = metric_tensor.volume(mt)
    # print(vol)
    # mt = metric_tensor.load('./tensor_cnd.npy')
    # vol = metric_tensor.volume(mt)
    # print(vol)