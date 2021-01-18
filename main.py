import argparse
import concurrent.futures
import json
import os
import platform
import subprocess

import torch

import A2C_Breakout
import A2C_QBert
import DDPG_AerisTargetNavigate
import DDPG_Ant
import DDPG_FetchReach
import DDPG_HalfCheetah
import DDPG_Hopper
import DDPG_LunarLander
import DDPG_MountainCar
import DDPG_Noisy_AerisTargetNavigate
import DDPG_Noisy_Ant
import DDPG_Noisy_HalfCheetah
import DDPG_Noisy_Hopper
import DDPG_Noisy_LunarLander
import DDPG_Noisy_Reacher
import DDPG_Reacher
import PPO_CartPole
import PPO_QBert
import PPO_Solaris
import PPO_Zelda
from utils.Config import Config


def set_env_class(env, noisy):
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
        if noisy:
            env_class = DDPG_Noisy_LunarLander
        else:
            env_class = DDPG_LunarLander
    if env == 'pong':
        pass
    if env == 'half_cheetah':
        if noisy:
            env_class = DDPG_Noisy_HalfCheetah
        else:
            env_class = DDPG_HalfCheetah
    if env == 'hopper':
        if noisy:
            env_class = DDPG_Noisy_Hopper
        else:
            env_class = DDPG_Hopper
    if env == 'ant':
        if noisy:
            env_class = DDPG_Noisy_Ant
        else:
            env_class = DDPG_Ant
    if env == 'reacher':
        if noisy:
            env_class = DDPG_Noisy_Reacher
        else:
            env_class = DDPG_Reacher
    if env == 'aeris_navigate':
        if noisy:
            env_class = DDPG_Noisy_AerisTargetNavigate
        else:
            env_class = DDPG_AerisTargetNavigate
    if env == 'zelda':
        env_class = PPO_Zelda

    return env_class


def run(env, experiment, id):
    print('Starting experiment {0}'.format(id + experiment.shift))

    env_class = set_env_class(env, experiment.noisy)

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


def write_command_file(args, experiment):
    thread_per_env = max(torch.get_num_threads() // experiment.trials, 1)
    if platform.system() == 'Windows':
        file = open("run.bat", "w")
        file.write('set OMP_NUM_THREADS={0}\n'.format(thread_per_env))
        for i in range(experiment.trials):
            file.write('start "" python main.py --env {0} --config {1} -t -s {2} \n'.format(args.env, args.config, i))
        file.close()

    if platform.system() == 'Linux':
        file = open("run.sh", "w")
        for i in range(experiment.trials):
            file.write('OMP_NUM_THREADS={0} python main.py --env {1} --config {2} -t -s {3} & \n'.format(thread_per_env, args.env, args.config, i))
        file.close()


def run_command_file():
    if platform.system() == 'Windows':
        subprocess.call([r'run.bat'])
        if os.path.exists('run.bat'):
            os.remove('run.bat')
    if platform.system() == 'Linux':
        subprocess.call([r'run.sh'])
        if os.path.exists('run.sh'):
            os.remove('run.sh')

if __name__ == '__main__':
    print(platform.system())
    print(torch.__config__.show())
    print(torch.__config__.parallel_info())
    parser = argparse.ArgumentParser(description='Motivation models learning platform.')

    parser.add_argument('--env', type=str, help='environment name')
    parser.add_argument('--config', type=int, help='id of config')
    parser.add_argument('--device', type=str, help='device type', default='cpu')
    parser.add_argument('-s', '--shift', type=int, help='shift result id', default=0)
    parser.add_argument('-p', '--parallel', action="store_true", help='run envs in parallel')
    parser.add_argument('-t', '--thread', action="store_true", help='do not use: technical parameter for parallel run')

    args = parser.parse_args()

    with open('./config.json') as f:
        config = json.load(f)

    experiment = Config(config[args.env][str(args.config)], "{0}_{1}".format(args.env, str(args.config)))
    experiment.device = args.device
    experiment.shift = args.shift
    if args.thread:
        experiment.trials = 1

    if args.parallel:
        write_command_file(args, experiment)
        run_command_file()
    else:
        for i in range(experiment.trials):
            run(args.env, experiment, i)
