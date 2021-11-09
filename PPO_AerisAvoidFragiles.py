import PPO_AerisEnv
from agents.PPOAerisAgent import PPOAerisAgent


def run_baseline(config, i):
    PPO_AerisEnv.run_baseline('AvoidFragiles-v0', config, i, PPOAerisAgent)


def run_rnd_model(config, i):
    PPO_AerisEnv.run_rnd_model('AvoidFragiles-v0', config, i)
