import DDPG_PyBulletEnv

env_name = 'HalfCheetahBulletEnv-v0'


def run_baseline(config, i):
    DDPG_PyBulletEnv.run_baseline(env_name, config, i)


def run_forward_model(config, i):
    DDPG_PyBulletEnv.run_forward_model(env_name, config, i)


def run_forward_model_encoder(config, i):
    DDPG_PyBulletEnv.run_forward_model_encoder(env_name, config, i)


def run_inverse_model(config, i):
    DDPG_PyBulletEnv.run_inverse_model(env_name, config, i)


def run_forward_inverse_model(config, i):
    DDPG_PyBulletEnv.run_forward_inverse_model(env_name, config, i)


def run_m2_model(config, i):
    DDPG_PyBulletEnv.run_m2_model(env_name, config, i)


def run_m2s_model(config, i):
    DDPG_PyBulletEnv.run_m2s_model(env_name, config, i)


def run_rnd_model(config, i):
    DDPG_PyBulletEnv.run_rnd_model(env_name, config, i)


def run_qrnd_model(config, i):
    DDPG_PyBulletEnv.run_qrnd_model(env_name, config, i)


def run_dop_model(config, i):
    DDPG_PyBulletEnv.run_dop_model(env_name, config, i)


def run_metalearner_model(config, i):
    DDPG_PyBulletEnv.run_metalearner_model(env_name, config, i)


def run_metalearner_rnd_model(config, i):
    DDPG_PyBulletEnv.run_metalearner_rnd_model(env_name, config, i)
