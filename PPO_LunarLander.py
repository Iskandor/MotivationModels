import gym

from algorithms.PPO import PPO
from experiment.ppo_experiment import ExperimentPPO
from modules.PPO_Modules import PPONetwork, HEAD


def run_baseline(config, i):
    env = gym.make('LunarLanderContinuous-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    experiment = ExperimentPPO('LunarLanderContinuous-v2', env, config)

    agent = PPO(PPONetwork(state_dim, action_dim, config, HEAD.continuous),
                config.lr, config.actor_loss_weight, config.critic_loss_weight,
                config.batch_size, config.trajectory_size,
                config.beta, config.gamma)
    experiment.run_baseline(agent, i)

    env.close()
