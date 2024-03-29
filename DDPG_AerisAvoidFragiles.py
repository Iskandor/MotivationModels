import DDPG_AerisEnv

env_name = 'AvoidFragiles-v0'


def run_baseline(config, i):
    DDPG_AerisEnv.run_baseline(env_name, config, i)


def run_forward_model(config, i):
    DDPG_AerisEnv.run_forward_model(env_name, config, i)


def run_forward_model_encoder(config, i):
    DDPG_AerisEnv.run_forward_model_encoder(env_name, config, i)


def run_inverse_model(config, i):
    DDPG_AerisEnv.run_inverse_model(env_name, config, i)


def run_rnd_model(config, i):
    DDPG_AerisEnv.run_rnd_model(env_name, config, i)


def run_qrnd_model(config, i):
    DDPG_AerisEnv.run_qrnd_model(env_name, config, i)


def run_vdop_model(config, i):
    DDPG_AerisEnv.run_vdop_model(env_name, config, i)


def run_dop_model(config, i):
    DDPG_AerisEnv.run_dop_model(env_name, config, i)


def run_dop2_model(config, i):
    DDPG_AerisEnv.run_dop2_model(env_name, config, i)


def run_dop2q_model(config, i):
    DDPG_AerisEnv.run_dop2q_model(env_name, config, i)


def run_dop3_model(config, i):
    DDPG_AerisEnv.run_dop3_model(env_name, config, i)


def run_dop_ref_model(config, i):
    DDPG_AerisEnv.run_dop_ref_model(env_name, config, i)


def run_metalearner_model(config, i):
    DDPG_AerisEnv.run_metalearner_model(env_name, config, i)


def run_metalearner_rnd_model(config, i):
    DDPG_AerisEnv.run_metalearner_rnd_model(env_name, config, i)
