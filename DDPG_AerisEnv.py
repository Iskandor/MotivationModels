import gym
import gym_aeris.envs

from agents.DDPGAerisAgent import DDPGAerisAgent, DDPGAerisForwardModelAgent, DDPGAerisForwardModelEncoderAgent, DDPGAerisInverseModelAgent, DDPGAerisRNDModelAgent, DDPGAerisForwardInverseModelAgent, \
    DDPGAerisGatedMetacriticModelAgent, DDPGAerisMetaCriticRNDModelAgent, DDPGAerisQRNDModelAgent, DDPGAerisDOPAgent, DDPGAerisDOPRefAgent, \
    DDPGAerisDOPV2QAgent, DDPGAerisDOPV2Agent, DDPGAerisDOPV3Agent, DDPGAerisVanillaDOPAgent
from experiment.ddpg_experiment import ExperimentDDPG


def create_env(env_id):
    env = None
    if env_id == 'AvoidFragiles-v0':
        env = gym_aeris.envs.AvoidFragilesEnv()
    if env_id == 'AvoidHazards-v0':
        env = gym_aeris.envs.AvoidHazardsEnv()
    if env_id == 'TargetNavigate-v0':
        env = gym_aeris.envs.TargetNavigateEnv()
    if env_id == 'FoodGathering-v0':
        env = gym_aeris.envs.FoodGatheringEnv()

    # GridTargetSearchAEnv-v0
    # GridTargetSearchBEnv-v0

    return env


def run_baseline(env_name, config, trial):
    env = create_env(env_name)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG(env_name, env, config)

    agent = DDPGAerisAgent(state_dim, action_dim, config)
    experiment.run_baseline(agent, trial)

    env.close()


def run_forward_model(env_name, config, trial):
    env = create_env(env_name)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG(env_name, env, config)

    agent = DDPGAerisForwardModelAgent(state_dim, action_dim, config)
    experiment.run_forward_model(agent, trial)

    env.close()


def run_forward_model_encoder(env_name, config, trial):
    env = create_env(env_name)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG(env_name, env, config)

    agent = DDPGAerisForwardModelEncoderAgent(state_dim, action_dim, config)
    experiment.run_forward_model_encoder(agent, trial)

    env.close()


def run_inverse_model(env_name, config, trial):
    env = create_env(env_name)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG(env_name, env, config)

    agent = DDPGAerisInverseModelAgent(state_dim, action_dim, config)
    experiment.run_inverse_model(agent, trial)

    env.close()


def run_forward_inverse_model(env_name, config, trial):
    env = create_env(env_name)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG(env_name, env, config)

    agent = DDPGAerisForwardInverseModelAgent(state_dim, action_dim, config)
    experiment.run_forward_inverse_model(agent, trial)

    env.close()


def run_rnd_model(env_name, config, i):
    env = create_env(env_name)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG(env_name, env, config)

    agent = DDPGAerisRNDModelAgent(state_dim, action_dim, config)
    experiment.run_rnd_model(agent, i)

    env.close()


def run_qrnd_model(env_name, config, i):
    env = create_env(env_name)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG(env_name, env, config)

    agent = DDPGAerisQRNDModelAgent(state_dim, action_dim, config)
    experiment.run_qrnd_model(agent, i)

    env.close()


def run_vdop_model(env_name, config, i):
    env = create_env(env_name)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG(env_name, env, config)

    agent = DDPGAerisVanillaDOPAgent(state_dim, action_dim, config)
    experiment.run_vdop_model(agent, i)

    env.close()


def run_dop_model(env_name, config, i):
    env = create_env(env_name)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG(env_name, env, config)

    agent = DDPGAerisDOPAgent(state_dim, action_dim, config)
    experiment.run_dop_model(agent, i)

    env.close()


def run_dop2_model(env_name, config, i):
    env = create_env(env_name)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG(env_name, env, config)

    agent = DDPGAerisDOPV2Agent(state_dim, action_dim, config)
    experiment.run_dop2_model(agent, i)

    env.close()


def run_dop2q_model(env_name, config, i):
    env = create_env(env_name)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG(env_name, env, config)

    agent = DDPGAerisDOPV2QAgent(state_dim, action_dim, config)
    experiment.run_dop2_model(agent, i)

    env.close()


def run_dop3_model(env_name, config, i):
    env = create_env(env_name)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG(env_name, env, config)

    agent = DDPGAerisDOPV3Agent(state_dim, action_dim, config)
    experiment.run_dop3_model(agent, i)

    env.close()


def run_dop_ref_model(env_name, config, i):
    env = create_env(env_name)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG(env_name, env, config)

    agent = DDPGAerisDOPRefAgent(state_dim, action_dim, config)
    experiment.run_baseline(agent, i)

    env.close()


def run_metalearner_model(env_name, config, trial):
    env = create_env(env_name)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG(env_name, env, config)

    agent = DDPGAerisGatedMetacriticModelAgent(state_dim, action_dim, config)
    experiment.run_metalearner_model(agent, trial)

    env.close()


def run_metalearner_rnd_model(env_name, config, trial):
    env = create_env(env_name)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    experiment = ExperimentDDPG(env_name, env, config)

    agent = DDPGAerisMetaCriticRNDModelAgent(state_dim, action_dim, config)
    experiment.run_metalearner_rnd_model(agent, trial)

    env.close()
