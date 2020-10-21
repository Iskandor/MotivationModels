import argparse

import A2C_Breakout
import A2C_CartPole
import A2C_QBert
import DDPG_FetchReach
import DDPG_HalfCheetah
import DDPG_LunarLander
import DDPG_MountainCar
#import PPO_Go
#import PPO_Chess
import DQN_CartPole
import PPO_CartPole
import PPO_QBert
from utils.Config import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Motivation models learning platform.')

    parser.add_argument('--env', type=str, choices=['half_cheetah', 'mountain_car', 'lunar_lander', 'fetch_reach', 'go', 'chess', 'pong', 'cart_pole', 'breakout', 'qbert'])
    parser.add_argument('--config', type=str, help='path to config file')

    args = parser.parse_args()

    if args.config is not None:
        config = Config.parse_config(args.config)

    if args.env == "qbert":
        if config.model == 'baseline':
            # A2C_QBert.run_baseline(config)
            PPO_QBert.run_baseline(config)
        if config.model == 'fm':
            PPO_QBert.run_icm(config)
    if args.env == "breakout":
        if config.model == 'baseline':
            A2C_Breakout.run_baseline(config)
    if args.env == "cart_pole":
        if config.model == 'baseline':
            # A2C_CartPole.run_baseline(config)
            # DQN_CartPole.run()
            PPO_CartPole.run_baseline(config)
    if args.env == 'mountain_car':
        if args.model == 'baseline':
            DDPG_MountainCar.run_baseline(int(args.trials), int(args.episodes))
        if args.model == 'fm':
            DDPG_MountainCar.run_forward_model(int(args.trials), int(args.episodes))
        if args.model == 'su':
            DDPG_MountainCar.run_metalearner_model(int(args.trials), int(args.episodes))
    if args.env == 'fetch_reach':
        if args.model == 'baseline':
            DDPG_FetchReach.run_baseline(args)
        if args.model == 'fm':
            DDPG_FetchReach.run_forward_model(int(args.trials), int(args.episodes), int(args.batch_size), int(args.memory_size))
    if args.env == 'go':
        pass
        #PPO_Go.run_baseline(int(args.trials), int(args.episodes), int(args.batch_size), int(args.memory_size))
    if args.env == 'chess':
        pass
        #PPO_Chess.run_baseline(int(args.trials), int(args.episodes))
    if args.env == 'lunar_lander':
        if config.model == 'baseline':
            DDPG_LunarLander.run_baseline(config)
        if config.model == 'fm':
            DDPG_LunarLander.run_forward_model(config)
        if config.model == 's':
            DDPG_LunarLander.run_surprise_model(config)
        if config.model == 'su':
            DDPG_LunarLander.run_metalearner_model(config)
    if args.env == 'pong':
        if args.model == 'baseline':
            pass
    if args.env == 'half_cheetah':
        if config.model == 'baseline':
            DDPG_HalfCheetah.run_baseline(config)
        if config.model == 'fm':
            DDPG_HalfCheetah.run_forward_model(config)
        if config.model == 's':
            DDPG_HalfCheetah.run_surprise_model(args)
        if config.model == 'su':
            DDPG_HalfCheetah.run_metalearner_model(config)
        if config.model == 'm3':
            DDPG_HalfCheetah.run_m3_model(config)

    # DQN_FrozenLake.run()
    # DQN_CartPole.run()
