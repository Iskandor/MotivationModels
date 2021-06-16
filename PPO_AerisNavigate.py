import gym_aeris

import PPO_AerisEnv


def run_baseline(config, i):
    PPO_AerisEnv.run_baseline('TargetNavigate-v0', gym_aeris.envs.TargetNavigateEnv(), config, i)


def run_rnd_model(config, i):
    PPO_AerisEnv.run_rnd_model('TargetNavigate-v0', gym_aeris.envs.TargetNavigateEnv(), config, i)


def run_dop_model(config, i):
    PPO_AerisEnv.run_dop_model('TargetNavigate-v0', gym_aeris.envs.TargetNavigateEnv(), config, i)
