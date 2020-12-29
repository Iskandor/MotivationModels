import argparse
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

if __name__ == '__main__':
    print(torch.__config__.show())
    print(torch.__config__.parallel_info())
    parser = argparse.ArgumentParser(description='Motivation models learning platform.')

    parser.add_argument('--env', type=str, help='environment name')
    parser.add_argument('--config', type=int, help='id of config')
    parser.add_argument('--device', type=str, help='device type', default='cpu')
    parser.add_argument('-s', '--shift', type=int, help='shift result id', default=0)

    args = parser.parse_args()

    with open('./config.json') as f:
        config = json.load(f)

    experiment = Config(config[args.env][str(args.config)], "{0}_{1}".format(args.env, str(args.config)))
    experiment.device = args.device
    experiment.shift = args.shift

    if args.env == "solaris":

        if experiment.model == 'baseline':
            PPO_Solaris.run_baseline(experiment)
        if experiment.model == 'fm':
            PPO_Solaris.run_icm(experiment)
    if args.env == "qbert":
        if experiment.model == 'baseline':
            # A2C_QBert.run_baseline(experiment)
            PPO_QBert.run_baseline(experiment)
        if experiment.model == 'fm':
            PPO_QBert.run_icm(experiment)
    if args.env == "breakout":
        if experiment.model == 'baseline':
            A2C_Breakout.run_baseline(experiment)
    if args.env == "cart_pole":
        if experiment.model == 'baseline':
            # A2C_CartPole.run_baseline(experiment)
            # DQN_CartPole.run()
            PPO_CartPole.run_baseline(experiment)
    if args.env == 'mountain_car':
        if args.model == 'baseline':
            DDPG_MountainCar.run_baseline(int(args.trials), int(args.episodes))
        if args.model == 'fm':
            DDPG_MountainCar.run_forward_model(int(args.trials), int(args.episodes))
        if args.model == 'su':
            DDPG_MountainCar.run_metalearner_model(int(args.trials), int(args.episodes))
    if args.env == 'fetch_reach':
        if experiment.model == 'baseline':
            DDPG_FetchReach.run_baseline(experiment)
        if experiment.model == 'fm':
            pass
            # DDPG_FetchReach.run_forward_model(int(args.trials), int(args.episodes), int(args.batch_size), int(args.memory_size))
    if args.env == 'lunar_lander':
        if experiment.model == 'baseline':
            DDPG_Noisy_LunarLander.run_baseline(experiment)
        if experiment.model == 'fm':
            DDPG_Noisy_LunarLander.run_forward_model(experiment)
        if experiment.model == 's':
            DDPG_Noisy_LunarLander.run_metalearner_model(experiment)
        if experiment.model == 'su':
            DDPG_Noisy_LunarLander.run_metalearner_model(experiment)
    if args.env == 'pong':
        if args.model == 'baseline':
            pass
    if args.env == 'half_cheetah':
        if experiment.model == 'baseline':
            DDPG_Noisy_HalfCheetah.run_baseline(experiment)
        if experiment.model == 'fm':
            DDPG_Noisy_HalfCheetah.run_forward_model(experiment)
        if experiment.model == 'vfm':
            DDPG_Noisy_HalfCheetah.run_vae_forward_model(experiment)
        if experiment.model == 's':
            DDPG_Noisy_HalfCheetah.run_metalearner_model(experiment)
        if experiment.model == 'su':
            DDPG_Noisy_HalfCheetah.run_metalearner_model(experiment)
        if experiment.model == 'm3':
            pass
            # DDPG_Noisy_HalfCheetah.run_m3_model(experiment)
    if args.env == 'hopper':
        if experiment.model == 'baseline':
            DDPG_Noisy_Hopper.run_baseline(experiment)
        if experiment.model == 'fm':
            DDPG_Noisy_Hopper.run_forward_model(experiment)
        if experiment.model == 's':
            DDPG_Noisy_Hopper.run_metalearner_model(experiment)
        if experiment.model == 'su':
            DDPG_Noisy_Hopper.run_metalearner_model(experiment)
    if args.env == 'ant':
        if experiment.model == 'baseline':
            DDPG_Noisy_Ant.run_baseline(experiment)
        if experiment.model == 'fm':
            DDPG_Noisy_Ant.run_forward_model(experiment)
        if experiment.model == 's':
            DDPG_Noisy_Ant.run_metalearner_model(experiment)
        if experiment.model == 'su':
            DDPG_Noisy_Ant.run_metalearner_model(experiment)
    if args.env == 'reacher':
        if experiment.model == 'baseline':
            DDPG_Noisy_Reacher.run_baseline(experiment)
        if experiment.model == 'fm':
            DDPG_Noisy_Reacher.run_forward_model(experiment)
        if experiment.model == 'su':
            DDPG_Noisy_Reacher.run_metalearner_model(experiment)
    if args.env == 'aeris_navigate':
        if experiment.model == 'baseline':
            DDPG_Noisy_AerisTargetNavigate.run_baseline(experiment)
        if experiment.model == 'fm':
            DDPG_Noisy_AerisTargetNavigate.run_forward_model(experiment)
        if experiment.model == 'su':
            DDPG_Noisy_AerisTargetNavigate.run_metalearner_model(experiment)

    if args.env == 'zelda':
        if experiment.model == 'baseline':
            PPO_Zelda.run_baseline(experiment)

    # DQN_FrozenLake.run()
    # DQN_CartPole.run()
