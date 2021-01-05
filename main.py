import argparse
import concurrent.futures
import json

import torch

import A2C_Breakout
import A2C_QBert
import DDPG_FetchReach
import DDPG_HalfCheetah
import DDPG_LunarLander
import DDPG_MountainCar
# import PPO_Go
# import PPO_Chess
import DDPG_Noisy_AerisTargetNavigate
import DDPG_Noisy_Ant
import DDPG_Noisy_HalfCheetah
import DDPG_Noisy_Hopper
import DDPG_Noisy_LunarLander
import DDPG_Noisy_Reacher
import PPO_CartPole
import PPO_QBert
import PPO_Solaris
import PPO_Zelda
from utils.Config import Config


def set_env_class(env):
    env_class = None

    if env == "solaris":
        env_class = PPO_Solaris
    if env == "qbert":
        env_class = PPO_QBert
    if env == "breakout":
        env_class = A2C_Breakout
    if env == "cart_pole":
        env_class = PPO_CartPole
    if env == 'mountain_car':
        env_class = DDPG_MountainCar
    if env == 'fetch_reach':
        env_class = DDPG_FetchReach
    if env == 'lunar_lander':
        env_class = DDPG_Noisy_LunarLander
    if env == 'pong':
        pass
    if env == 'half_cheetah':
        env_class = DDPG_Noisy_HalfCheetah
    if env == 'hopper':
        env_class = DDPG_Noisy_Hopper
    if env == 'ant':
        env_class = DDPG_Noisy_Ant
    if env == 'reacher':
        env_class = DDPG_Noisy_Reacher
    if env == 'aeris_navigate':
        env_class = DDPG_Noisy_AerisTargetNavigate
    if env == 'zelda':
        env_class = PPO_Zelda

    return env_class


def run(env, experiment, id):
    print('Starting experiment {0}'.format(id))

    env_class = set_env_class(env)

    if experiment.model == 'baseline':
        env_class.run_baseline(experiment, id)
    if experiment.model == 'fm':
        env_class.run_forward_model(experiment, id)
    if experiment.model == 'rfm':
        env_class.run_residual_forward_model(experiment, id)
    if experiment.model == 'vfm':
        env_class.run_vae_forward_model(experiment, id)
    if experiment.model == 's':
        env_class.run_metalearner_model(experiment, id)
    if experiment.model == 'su':
        env_class.run_metalearner_model(experiment, id)
    if experiment.model == 'm3':
        env_class.run_m3_model(experiment, id)


if __name__ == '__main__':
    print(torch.__config__.show())
    print(torch.__config__.parallel_info())
    parser = argparse.ArgumentParser(description='Motivation models learning platform.')

    parser.add_argument('--env', type=str, help='environment name')
    parser.add_argument('--config', type=int, help='id of config')
    parser.add_argument('--device', type=str, help='device type', default='cpu')
    parser.add_argument('-s', '--shift', type=int, help='shift result id', default=0)
    parser.add_argument('-p', '--parallel', action="store_true", help='run envs in parallel')

    args = parser.parse_args()

    with open('./config.json') as f:
        config = json.load(f)

    experiment = Config(config[args.env][str(args.config)], "{0}_{1}".format(args.env, str(args.config)))
    experiment.device = args.device
    experiment.shift = args.shift

    if args.parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=experiment.trials) as executor:
            executor.map(run, [args.env] * experiment.trials, [experiment] * experiment.trials, range(experiment.trials))
    else:
        for i in range(experiment.trials):
            run(args.env, experiment, i)
