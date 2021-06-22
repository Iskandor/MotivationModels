import PPO_AerisEnv


def run_baseline(config, i):
    PPO_AerisEnv.run_baseline('AvoidFragiles-v0', config, i)


def run_rnd_model(config, i):
    PPO_AerisEnv.run_rnd_model('AvoidFragiles-v0', config, i)
