import gym
import pybullet_envs
from algorithms.PPOContinuous import PPOContinuous
from experiment.ppo_continuous_experiment import ExperimentPPOContinuous
from modules.PPO_Modules import ContinuousPPONetwork


def run_baseline(config, i):
    env = gym.make('ReacherBulletEnv-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentPPOContinuous('ReacherBulletEnv-v0', env, config)

    network = ContinuousPPONetwork(state_dim, action_dim, config)
    agent = PPOContinuous(network, config.lr, config.actor_loss_weight, config.critic_loss_weight, config.batch_size, config.trajectory_size, config.beta, config.gamma)
    experiment.run_baseline(agent, i)

    env.close()