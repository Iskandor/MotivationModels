import argparse

import DDPG_FetchReach
import DDPG_MountainCar

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Motivation models learning platform.')

    parser.add_argument('--env', help='[mountain_car,fetch_reach]')
    parser.add_argument('--model', help='[baseline,fm,su]')
    parser.add_argument('--trials', help='No. trials')
    parser.add_argument('--episodes', help='No. episodes')
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
            DDPG_FetchReach.run_baseline(int(args.trials), int(args.episodes), int(args.memory_size))
        if args.model == 'fm':
            DDPG_FetchReach.run_forward_model(int(args.trials), int(args.episodes), int(args.memory_size))

    # DQN_FrozenLake.run()
    # DQN_CartPole.run()
