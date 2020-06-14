import argparse

import DDPG_FetchReach
import DDPG_LunarLander
import DDPG_MountainCar
import PPO_Chess
import PPO_Go

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Motivation models learning platform.')

    parser.add_argument('--env', help='[mountain_car,fetch_reach,go]')
    parser.add_argument('--model', help='[baseline,fm,su]')
    parser.add_argument('--trials', help='No. trials')
    parser.add_argument('--episodes', help='No. episodes')
    parser.add_argument('--batch_size', default=64, help='Minibatch size')
    parser.add_argument('--memory_size', default=10000, help='Size of memory buffer')

    args = parser.parse_args()

    if args.env == 'mountain_car':
        if args.model == 'baseline':
            DDPG_MountainCar.run_baseline(int(args.trials), int(args.episodes))
        if args.model == 'fm':
            DDPG_MountainCar.run_forward_model(int(args.trials), int(args.episodes))
        if args.model == 'su':
            DDPG_MountainCar.run_metalearner_model(int(args.trials), int(args.episodes))
    if args.env == 'fetch_reach':
        if args.model == 'baseline':
            DDPG_FetchReach.run_baseline(int(args.trials), int(args.episodes), int(args.batch_size), int(args.memory_size))
        if args.model == 'fm':
            DDPG_FetchReach.run_forward_model(int(args.trials), int(args.episodes), int(args.batch_size), int(args.memory_size))
    if args.env == 'go':
        PPO_Go.run_baseline(int(args.trials), int(args.episodes), int(args.batch_size), int(args.memory_size))
    if args.env == 'chess':
        PPO_Chess.run_baseline(int(args.trials), int(args.episodes))
    if args.env == 'lunar_lander':
        DDPG_LunarLander.run_baseline(int(args.trials), int(args.episodes), int(args.batch_size), int(args.memory_size))

    # DQN_FrozenLake.run()
    # DQN_CartPole.run()
