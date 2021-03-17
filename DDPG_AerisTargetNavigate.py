import gym
import gym_aeris.envs

from agents.DDPGAgent import DDPGAerisAgent, DDPGAerisForwardModelAgent, DDPGAerisForwardModelEncoderAgent, DDPGAerisInverseModelAgent, DDPGAerisM2ModelAgent
from algorithms.DDPG import DDPG
from algorithms.ReplayBuffer import ExperienceReplayBuffer
from experiment.ddpg_experiment import ExperimentDDPG
from modules import ARCH
from modules.forward_models.RND_ForwardModel import RND_ForwardModel
from motivation.ForwardModelMotivation import ForwardModelMotivation
from motivation.MateCriticMotivation import MetaCriticMotivation


def run_baseline(config, i):
    env = gym_aeris.envs.TargetNavigateEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('TargetNavigate-v0', env, config)

    agent = DDPGAerisAgent(state_dim, action_dim, config)
    experiment.run_baseline(agent, i)

    env.close()


def run_forward_model(config, i):
    env = gym_aeris.envs.TargetNavigateEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('TargetNavigate-v0', env, config)

    agent = DDPGAerisForwardModelAgent(state_dim, action_dim, config)
    experiment.run_forward_model(agent, i)

    env.close()


def run_forward_model_encoder(config, i):
    env = gym_aeris.envs.TargetNavigateEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('TargetNavigate-v0', env, config)

    agent = DDPGAerisForwardModelEncoderAgent(state_dim, action_dim, config)
    experiment.run_forward_model_encoder(agent, i)

    env.close()


def run_inverse_model(config, i):
    env = gym_aeris.envs.TargetNavigateEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('TargetNavigate-v0', env, config)

    agent = DDPGAerisInverseModelAgent(state_dim, action_dim, config)
    experiment.run_inverse_model(agent, i)

    env.close()


def run_m2_model(config, i):
    env = gym_aeris.envs.TargetNavigateEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('TargetNavigate-v0', env, config)

    agent = DDPGAerisM2ModelAgent(state_dim, action_dim, config)
    experiment.run_m2_model(agent, i)

    env.close()

def run_rnd_forward_model(config, i):
    env = gym_aeris.envs.TargetNavigateEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('TargetNavigate-v0', env, config)

    actor = Actor(state_dim, action_dim, config)
    critic = Critic(state_dim, action_dim, config)
    memory = ExperienceReplayBuffer(config.memory_size)

    agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)

    if hasattr(config, 'forward_model_batch_size'):
        forward_model = ForwardModelMotivation(RND_ForwardModel(state_dim, action_dim, config, ARCH.aeris), config.forward_model_lr, config.forward_model_eta,
                                               config.forward_model_variant, 1000 * 10,
                                               memory, config.forward_model_batch_size)
    else:
        forward_model = ForwardModelMotivation(RND_ForwardModel(state_dim, action_dim, config, ARCH.aeris), config.forward_model_lr, config.forward_model_eta,
                                               config.forward_model_variant, 1000 * 10)

    agent.add_motivation_module(forward_model)

    experiment.run_forward_model(agent, i)

    env.close()


def run_metalearner_model(config, i):
    env = gym_aeris.envs.TargetNavigateEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('TargetNavigate-v0', env, config)

    actor = Actor(state_dim, action_dim, config)
    critic = Critic(state_dim, action_dim, config)
    memory = ExperienceReplayBuffer(config.memory_size)

    agent = DDPG(actor, critic, config.actor_lr, config.critic_lr, config.gamma, config.tau, memory, config.batch_size)

    if hasattr(config, 'metacritic_batch_size'):
        metacritic = MetaCriticMotivation(MetaCritic(state_dim, action_dim, config), config.metacritic_lr, config.metacritic_eta,
                                          memory, config.metacritic_batch_size)
    else:
        metacritic = MetaCriticMotivation(MetaCritic(state_dim, action_dim, config), config.metacritic_lr, config.metacritic_variant, config.metacritic_eta)

    agent.add_motivation_module(metacritic)

    experiment.run_metalearner_model(agent, i)

    env.close()
