import gym
import gym_aeris.envs

from agents.DDPGAgent import DDPGAerisAgent, DDPGAerisForwardModelAgent, DDPGAerisForwardModelEncoderAgent, DDPGAerisInverseModelAgent, DDPGAerisM2ModelAgent, DDPGAerisForwardInverseModelAgent, \
    DDPGAerisGatedMetacriticModelAgent
from algorithms.DDPG import DDPG
from algorithms.ReplayBuffer import ExperienceReplayBuffer
from experiment.ddpg_experiment import ExperimentDDPG
from modules import ARCH
from modules.forward_models.RND_ForwardModel import RND_ForwardModel
from motivation.ForwardModelMotivation import ForwardModelMotivation
from motivation.MetaCriticMotivation import MetaCriticMotivation


def run_baseline(config, i):
    env = gym_aeris.envs.AvoidHazardsEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AvoidHazards-v0', env, config)

    agent = DDPGAerisAgent(state_dim, action_dim, config)
    experiment.run_baseline(agent, i)

    env.close()


def run_forward_model(config, i):
    env = gym_aeris.envs.AvoidHazardsEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AvoidHazards-v0', env, config)

    agent = DDPGAerisForwardModelAgent(state_dim, action_dim, config)
    experiment.run_forward_model(agent, i)

    env.close()


def run_forward_model_encoder(config, i):
    env = gym_aeris.envs.AvoidHazardsEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AvoidHazards-v0', env, config)

    agent = DDPGAerisForwardModelEncoderAgent(state_dim, action_dim, config)
    experiment.run_forward_model_encoder(agent, i)

    env.close()


def run_inverse_model(config, i):
    env = gym_aeris.envs.AvoidHazardsEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AvoidHazards-v0', env, config)

    agent = DDPGAerisInverseModelAgent(state_dim, action_dim, config)
    experiment.run_inverse_model(agent, i)

    env.close()


def run_forward_inverse_model(config, i):
    env = gym_aeris.envs.AvoidHazardsEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AvoidHazards-v0', env, config)

    agent = DDPGAerisForwardInverseModelAgent(state_dim, action_dim, config)
    experiment.run_forward_inverse_model(agent, i)

    env.close()


def run_m2_model(config, i):
    env = gym_aeris.envs.AvoidHazardsEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AvoidHazards-v0', env, config)

    agent = DDPGAerisM2ModelAgent(state_dim, action_dim, config)
    experiment.run_m2_model(agent, i)

    env.close()


def run_rnd_forward_model(config, i):
    env = gym_aeris.envs.AvoidHazardsEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AvoidHazards-v0', env, config)

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
    env = gym_aeris.envs.AvoidHazardsEnv()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG('AvoidHazards-v0', env, config)

    agent = DDPGAerisGatedMetacriticModelAgent(state_dim, action_dim, config)
    experiment.run_metalearner_model(agent, i)

    env.close()
