import PPO_AerisEnv
from agents.PPOAerisAgent import PPOAerisAgent, PPOAerisRNDAgent, PPOAerisDOPAgent, PPOAerisDOPRefAgent

env_name = 'AvoidHazards-v0'

def run_baseline(config, i):
    PPO_AerisEnv.run_baseline(env_name, config, i, PPOAerisAgent)


def run_rnd_model(config, i):
    PPO_AerisEnv.run_rnd_model(env_name, config, i, PPOAerisRNDAgent)


def run_dop_model(config, i):
    PPO_AerisEnv.run_dop_model(env_name, config, i, PPOAerisDOPAgent)


def run_dop_ref_model(config, i):
    PPO_AerisEnv.run_dop_ref_model(env_name, config, i, PPOAerisDOPRefAgent)
