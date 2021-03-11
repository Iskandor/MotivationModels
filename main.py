import argparse
import json
import os
import platform
import subprocess
import multiprocessing

import psutil
import ray
import torch

import A2C_Breakout
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
import PPO_LunarLander
import PPO_MidnightResistance
import PPO_MountainCar
import PPO_Pacman
import PPO_Pendulum
import PPO_QBert
import PPO_Solaris
from config import load_config_file
from config.Config import Config


def set_env_class_ddpg(env, experiment):
    env_class = None

    if env == 'mountain_car':
        env_class = DDPG_MountainCar
    if env == 'fetch_reach':
        env_class = DDPG_FetchReach
    if env == 'lunar_lander':
        if experiment.noisy:
            env_class = DDPG_Noisy_LunarLander
        else:
            env_class = DDPG_LunarLander
    if env == 'half_cheetah':
        if experiment.noisy:
            env_class = DDPG_Noisy_HalfCheetah
        else:
            env_class = DDPG_HalfCheetah
    if env == 'hopper':
        if experiment.noisy:
            env_class = DDPG_Noisy_Hopper
        else:
            env_class = DDPG_Hopper
    if env == 'ant':
        if experiment.noisy:
            env_class = DDPG_Noisy_Ant
        else:
            env_class = DDPG_Ant
    if env == 'reacher':
        if experiment.noisy:
            env_class = DDPG_Noisy_Reacher
        else:
            env_class = DDPG_Reacher
    if env == 'aeris_navigate':
        if experiment.noisy:
            env_class = DDPG_Noisy_AerisTargetNavigate
        else:
            env_class = DDPG_AerisTargetNavigate

    return env_class


def set_env_class_ppo(env):
    env_class = None

    if env == "solaris":
        env_class = PPO_Solaris
    if env == "qbert":
        env_class = PPO_QBert
    if env == "mspacman":
        env_class = PPO_Pacman
    if env == "midnight_resistance":
        env_class = PPO_MidnightResistance
    if env == "cart_pole":
        env_class = PPO_CartPole
    if env == 'mountain_car':
        env_class = PPO_MountainCar
    if env == 'pendulum':
        env_class = PPO_Pendulum
    if env == 'lunar_lander':
        env_class = PPO_LunarLander

    return env_class


def set_env_class_a2c(env):
    env_class = None

    if env == "breakout":
        env_class = A2C_Breakout

    return env_class


def set_env_class(algorithm, env, experiment):
    env_class = None

    if algorithm == 'a2c':
        env_class = set_env_class_a2c(env)
    if algorithm == 'ddpg':
        env_class = set_env_class_ddpg(env, experiment)
    if algorithm == 'ppo':
        env_class = set_env_class_ppo(env)

    return env_class


def run_parallel(args, experiment):
    thread_params = []
    for i in range(experiment.trials):
        thread_params.append((args.algorithm, args.env, experiment, i))

    ray.get([run_thread.remote(tp) for tp in thread_params])


@ray.remote
def run_thread(thread_params):
    algorithm, env, experiment, i = thread_params
    run(algorithm, env, experiment, i)


def run(algorithm, env, experiment, id):
    print('Starting experiment {0} on env {1} learning algorithm {2}'.format(id + experiment.shift, env, algorithm))

    env_class = set_env_class(algorithm, env, experiment)

    if experiment.model == 'baseline':
        env_class.run_baseline(experiment, id)
    if experiment.model == 'fm':
        env_class.run_forward_model(experiment, id)
    if experiment.model == 'rfm':
        env_class.run_residual_forward_model(experiment, id)
    if experiment.model == 'vfm':
        env_class.run_vae_forward_model(experiment, id)
    if experiment.model == 'rnd':
        env_class.run_rnd_forward_model(experiment, id)
    if experiment.model == 's':
        env_class.run_metalearner_model(experiment, id)
    if experiment.model == 'su':
        env_class.run_metalearner_model(experiment, id)
    if experiment.model == 'm2':
        env_class.run_m2_model(experiment, id)
    if experiment.model == 'm3':
        env_class.run_m3_model(experiment, id)


def write_command_file(args, experiment):
    print(multiprocessing.cpu_count())
    thread_per_env = max(multiprocessing.cpu_count() // experiment.trials, 1)
    if platform.system() == 'Windows':
        file = open("run.bat", "w")
        file.write('set OMP_NUM_THREADS={0}\n'.format(thread_per_env))
        for i in range(experiment.trials):
            file.write('start "" python main.py --env {0} --config {1} -t -s {2} \n'.format(args.env, args.config, i + args.shift))
        file.close()

    if platform.system() == 'Linux':
        file = open("run.sh", "w")
        for i in range(experiment.trials):
            file.write(
                'OMP_NUM_THREADS={0} python3 main.py --env {1} --config {2} -t -s {3} & \n'.format(thread_per_env, args.env, args.config, i + args.shift))
        file.close()


def run_command_file():
    if platform.system() == 'Windows':
        subprocess.call([r'run.bat'])
        if os.path.exists('run.bat'):
            os.remove('run.bat')
    if platform.system() == 'Linux':
        os.chmod('run.sh', 777)
        subprocess.run(['bash', './run.sh'])
        if os.path.exists('./run.sh'):
            os.remove('./run.sh')


if __name__ == '__main__':
    print(platform.system())
    print(torch.__config__.show())
    print(torch.__config__.parallel_info())
    parser = argparse.ArgumentParser(description='Motivation models learning platform.')

    if not os.path.exists('./models'):
        os.mkdir('./models')

    parser.add_argument('--env', type=str, help='environment name')
    parser.add_argument('-a', '--algorithm', type=str, help='training algorithm', choices=['ppo', 'ddpg', 'a2c', 'dqn'])
    parser.add_argument('--config', type=int, help='id of config')
    parser.add_argument('--device', type=str, help='device type', default='cpu')
    parser.add_argument('--gpus', help='device ids', default=None)
    parser.add_argument('--load', type=str, help='path to saved agent', default='')
    parser.add_argument('-s', '--shift', type=int, help='shift result id', default=0)
    parser.add_argument('-p', '--parallel', action="store_true", help='run envs in parallel')
    parser.add_argument('-t', '--thread', action="store_true", help='do not use: technical parameter for parallel run')

    args = parser.parse_args()
    if args.gpus:
        args.gpus = [int(s) for s in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
    config = load_config_file(args.algorithm)

    experiment = Config(config[args.env][str(args.config)], "{0}_{1}".format(args.env, str(args.config)))
    experiment.device = args.device
    experiment.gpus = args.gpus
    experiment.shift = args.shift

    if args.load != '':
        env_class = set_env_class(args.env, experiment)
        env_class.test(experiment, args.load)
    else:
        if args.thread:
            experiment.trials = 1

        if args.parallel:
            num_cpus = psutil.cpu_count(logical=True)
            print('Running parallel on {0} CPUs'.format(num_cpus))
            ray.init(num_cpus=num_cpus)
            torch.set_num_threads(num_cpus // experiment.trials)

            run_parallel(args, experiment)
            # write_command_file(args, experiment)
            # run_command_file()
        else:
            for i in range(experiment.trials):
                run(args.algorithm, args.env, experiment, i)
