import gym
import torch

from agents import TYPE
from agents.PPOAgent import PPOAtariAgent
from algorithms.PPO import PPO
from experiment.ppo_nenv_experiment import ExperimentNEnvPPO
from modules import ARCH
from experiment.ppo_experiment import ExperimentPPO
from modules.PPO_Modules import PPOAtariNetwork
from modules.forward_models.ForwardModel import ForwardModel
from motivation.ForwardModelMotivation import ForwardModelMotivation
from utils.AtariWrapper import WrapperAtari


def encode_state(state):
    return torch.tensor(state, dtype=torch.float32)


def test(config, path):
    env = WrapperAtari(gym.make('MsPacman-v0'))
    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    experiment = ExperimentPPO('MsPacman-v0', env, config)
    experiment.add_preprocess(encode_state)

    agent = PPOAtariAgent(input_shape, action_dim, config, TYPE.discrete)
    agent.load(path)
    experiment.test(agent)

    env.close()


def run_baseline(config, trial):
    env = WrapperAtari(gym.make('MsPacman-v0'))
    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    if config.n_env > 1:
        env_list = []
        print('Creating {0:d} environments'.format(config.n_env))
        for i in range(config.n_env):
            env_list.append(WrapperAtari(gym.make('MsPacman-v0')))

        print('Start training')
        experiment = ExperimentNEnvPPO('MsPacman-v0', env_list, config)
    else:
        experiment = ExperimentPPO('MsPacman-v0', env, config)
        experiment.add_preprocess(encode_state)

    agent = PPOAtariAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_baseline(agent, trial)

    env.close()

    if config.n_env > 1:
        for i in range(config.n_env):
            env_list[i].close()


def run_forward_model(config, trial):
    env = WrapperAtari(gym.make('MsPacman-v0'))
    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    if config.n_env > 1:
        env_list = []
        print('Creating {0:d} environments'.format(config.n_env))
        for i in range(config.n_env):
            env_list.append(WrapperAtari(gym.make('MsPacman-v0')))

        print('Start training')
        experiment = ExperimentNEnvPPO('MsPacman-v0', env_list, config)
    else:
        experiment = ExperimentPPO('MsPacman-v0', env, config)
        experiment.add_preprocess(encode_state)

    network = PPOAtariNetwork(input_shape, action_dim, config).to(config.device)

    forward_model = ForwardModelMotivation(ForwardModel(input_shape, action_dim, config, ARCH.atari).to(config.device), config.forward_model_lr, config.forward_model_eta,
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
