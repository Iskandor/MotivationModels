import PPO_AerisEnv
from agents.PPOAerisAgent import PPOAerisGridAgent


def run_baseline(config, i):
    PPO_AerisEnv.run_baseline('GridTargetSearchAEnv-v0', config, i, PPOAerisGridAgent)


def run_rnd_model(config, i):
    PPO_AerisEnv.run_rnd_model('GridTargetSearchAEnv-v0', config, i)
