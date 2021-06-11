import gym_aeris

import PPO_AerisEnv


def run_baseline(config, i):
    PPO_AerisEnv.run_baseline('AvoidHazards-v0', gym_aeris.envs.AvoidHazardsEnv(), config, i)
