import gym
import torch

from algorithms.PPO import PPO
from modules.PPO_Modules import AtariPPONetwork
from experiment.ppo_experiment import ExperimentPPO
from utils.AtariWrapper import AtariWrapper


def encode_state(state):
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0)


def run_baseline(config, i):
    env = AtariWrapper(gym.make('Qbert-v0'))
    input_shape = env.observation_space.shape
    action_dim = env.action_space.n

    experiment = ExperimentPPO('Qbert-v0', env, config)
    experiment.add_preprocess(encode_state)

    network = AtariPPONetwork(input_shape, action_dim, config).to(config.device)
    agent = PPO(network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size, config.beta, config.gamma,
                device=config.device)
    experiment.run_baseline(agent, i)

    env.close()
