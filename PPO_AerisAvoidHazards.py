import PPO_AerisEnv
from agents.PPOAerisAgent import PPOAerisAgent


def run_baseline(config, i):
    PPO_AerisEnv.run_baseline('AvoidHazards-v0', config, i, PPOAerisAgent)


def run_rnd_model(config, i):
    PPO_AerisEnv.run_rnd_model('AvoidHazards-v0', config, i)


def run_dop_model(config, i):
    PPO_AerisEnv.run_dop_model('AvoidHazards-v0', config, i)
