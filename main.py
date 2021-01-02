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
#import PPO_Go
#import PPO_Chess
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


def run(env, experiment, id):
    print('Starting experiment {0}'.format(id))
    if env == "solaris":
        if experiment.model == 'baseline':
            PPO_Solaris.run_baseline(experiment, id)
        if experiment.model == 'fm':
            PPO_Solaris.run_icm(experiment, id)
    if env == "qbert":
        if experiment.model == 'baseline':
            # A2C_QBert.run_baseline(experiment, id)
            PPO_QBert.run_baseline(experiment, id)
        if experiment.model == 'fm':
            PPO_QBert.run_icm(experiment, id)
    if env == "breakout":
        if experiment.model == 'baseline':
            A2C_Breakout.run_baseline(experiment, id)
    if env == "cart_pole":
        if experiment.model == 'baseline':
            # A2C_CartPole.run_baseline(experiment, id)
            # DQN_CartPole.run(, id)
            PPO_CartPole.run_baseline(experiment, id)
    if env == 'mountain_car':
        if args.model == 'baseline':
            DDPG_MountainCar.run_baseline(int(args.trials), int(args.episodes))
        if args.model == 'fm':
            DDPG_MountainCar.run_forward_model(int(args.trials), int(args.episodes))
        if args.model == 'su':
            DDPG_MountainCar.run_metalearner_model(int(args.trials), int(args.episodes))
    if env == 'fetch_reach':
        if experiment.model == 'baseline':
            DDPG_FetchReach.run_baseline(experiment, id)
        if experiment.model == 'fm':
            pass
    if env == 'lunar_lander':
        if experiment.model == 'baseline':
            DDPG_Noisy_LunarLander.run_baseline(experiment, id)
        if experiment.model == 'fm':
            DDPG_Noisy_LunarLander.run_forward_model(experiment, id)
        if experiment.model == 's':
            DDPG_Noisy_LunarLander.run_metalearner_model(experiment, id)
        if experiment.model == 'su':
            DDPG_Noisy_LunarLander.run_metalearner_model(experiment, id)
    if env == 'pong':
        if args.model == 'baseline':
            pass
    if env == 'half_cheetah':
        if experiment.model == 'baseline':
            DDPG_Noisy_HalfCheetah.run_baseline(experiment, id)
        if experiment.model == 'fm':
            DDPG_Noisy_HalfCheetah.run_forward_model(experiment, id)
        if experiment.model == 'vfm':
            DDPG_Noisy_HalfCheetah.run_vae_forward_model(experiment, id)
        if experiment.model == 's':
            DDPG_Noisy_HalfCheetah.run_metalearner_model(experiment, id)
        if experiment.model == 'su':
            DDPG_Noisy_HalfCheetah.run_metalearner_model(experiment, id)
        if experiment.model == 'm3':
            pass
            # DDPG_Noisy_HalfCheetah.run_m3_model(experiment, id)
    if env == 'hopper':
        if experiment.model == 'baseline':
            DDPG_Noisy_Hopper.run_baseline(experiment, id)
        if experiment.model == 'fm':
            DDPG_Noisy_Hopper.run_forward_model(experiment, id)
        if experiment.model == 's':
            DDPG_Noisy_Hopper.run_metalearner_model(experiment, id)
        if experiment.model == 'su':
            DDPG_Noisy_Hopper.run_metalearner_model(experiment, id)
    if env == 'ant':
        if experiment.model == 'baseline':
            DDPG_Noisy_Ant.run_baseline(experiment, id)
        if experiment.model == 'fm':
            DDPG_Noisy_Ant.run_forward_model(experiment, id)
        if experiment.model == 's':
            DDPG_Noisy_Ant.run_metalearner_model(experiment, id)
        if experiment.model == 'su':
            DDPG_Noisy_Ant.run_metalearner_model(experiment, id)
    if env == 'reacher':
        if experiment.model == 'baseline':
            DDPG_Noisy_Reacher.run_baseline(experiment, id)
        if experiment.model == 'fm':
            DDPG_Noisy_Reacher.run_forward_model(experiment, id)
        if experiment.model == 'su':
            DDPG_Noisy_Reacher.run_metalearner_model(experiment, id)
    if env == 'aeris_navigate':
        if experiment.model == 'baseline':
            DDPG_Noisy_AerisTargetNavigate.run_baseline(experiment, id)
        if experiment.model == 'fm':
            DDPG_Noisy_AerisTargetNavigate.run_forward_model(experiment, id)
        if experiment.model == 'su':
            DDPG_Noisy_AerisTargetNavigate.run_metalearner_model(experiment, id)

    if env == 'zelda':
        if experiment.model == 'baseline':
            PPO_Zelda.run_baseline(experiment, id)


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
