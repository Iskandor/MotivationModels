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
    env = WrapperAtari(gym.make('Pitfall-v0'))
    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    experiment = ExperimentPPO('Pitfall-v0', env, config)
    experiment.add_preprocess(encode_state)

    agent = PPOAtariAgent(input_shape, action_dim, config, TYPE.discrete)
    agent.load(path)
    experiment.test(agent)

    env.close()


def run_baseline(config, trial):
    env = WrapperAtari(gym.make('Pitfall-v0'))
    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    if config.n_env > 1:
        env_list = []
        print('Creating {0:d} environments'.format(config.n_env))
        for i in range(config.n_env):
            env_list.append(WrapperAtari(gym.make('Pitfall-v0')))

        print('Start training')
        experiment = ExperimentNEnvPPO('Pitfall-v0', env_list, config)
    else:
        experiment = ExperimentPPO('Pitfall-v0', env, config)
        experiment.add_preprocess(encode_state)

    agent = PPOAtariAgent(input_shape, action_dim, config, TYPE.discrete)
    experiment.run_baseline(agent, trial)

    env.close()

    if config.n_env > 1:
        for i in range(config.n_env):
            env_list[i].close()
