import gym
import retro
import torch
import numpy as np

from algorithms.PPO import PPO
from experiment.ppo_nenv_experiment import ExperimentNEnvPPO
from modules import ARCH
from modules.PPO_Modules import SegaPPONetwork
from experiment.ppo_experiment import ExperimentPPO
from modules.forward_models.ForwardModel import ForwardModel
from motivation.ForwardModelMotivation import ForwardModelMotivation


def encode_state(state):
    return torch.tensor(np.transpose(state, (2, 0, 1)), dtype=torch.float32).unsqueeze(0)


def test(config, path):
    env = retro.make(game='MidnightResistance-Genesis')
    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    experiment = ExperimentPPO('MidnightResistance-Genesis', env, config)
    experiment.add_preprocess(encode_state)

    network = AtariPPONetwork(input_shape, action_dim, config).to(config.device)
    agent = PPO(network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size, config.beta, config.gamma,
                device=config.device)
    agent.load(path)
    experiment.test(agent)

    env.close()


def run_baseline(config, trial):
    env = retro.make(game='MidnightResistance-Genesis')
    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    if config.n_env > 1:
        env_list = []
        print('Creating {0:d} environments'.format(config.n_env))
        for i in range(config.n_env):
            env_list.append(retro.make(game='MidnightResistance-Genesis'))

        print('Start training')
        experiment = ExperimentNEnvPPO('MidnightResistance-Genesis', env_list, config)
    else:
        experiment = ExperimentPPO('MidnightResistance-Genesis', env, config)
        experiment.add_preprocess(encode_state)

    network = SegaPPONetwork(input_shape, action_dim, config).to(config.device)
    agent = PPO(network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size, config.beta, config.gamma,
                device=config.device, n_env=config.n_env)
    experiment.run_baseline(agent, trial)

    env.close()

    if config.n_env > 1:
        for i in range(config.n_env):
            env_list[i].close()


def run_forward_model(config, trial):
    env = retro.make(game='MidnightResistance-Genesis')
    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    if config.n_env > 1:
        env_list = []
        print('Creating {0:d} environments'.format(config.n_env))
        for i in range(config.n_env):
            env_list.append(retro.make(game='MidnightResistance-Genesis'))

        print('Start training')
        experiment = ExperimentNEnvPPO('MidnightResistance-Genesis', env_list, config)
    else:
        experiment = ExperimentPPO('MidnightResistance-Genesis', env, config)
        experiment.add_preprocess(encode_state)

    network = SegaPPONetwork(input_shape, action_dim, config).to(config.device)

    forward_model = ForwardModelMotivation(ForwardModel(input_shape, action_dim, config, ARCH.atari).to(config.device), config.forward_model_lr,
                                           config.forward_model_eta,
                                           config.forward_model_variant, env.spec.max_episode_steps * 10,
                                           None, config.forward_model_batch_size, device=config.device)
    agent = PPO(network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size, config.beta, config.gamma,
                device=config.device, n_env=config.n_env)
    agent.add_motivation_module(forward_model)

    experiment.run_forward_model(agent, trial)

    env.close()

    if config.n_env > 1:
        for i in range(config.n_env):
            env_list[i].close()
