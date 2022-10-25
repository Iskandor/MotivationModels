import PPO_ProcgenGame

env_name = 'procgen-coinrun-v0'


def test(config, path):
    PPO_ProcgenGame.test(config, path, env_name)


def run_baseline(config, trial):
    PPO_ProcgenGame.run_baseline(config, trial, env_name)


def run_rnd_model(config, trial):
    PPO_ProcgenGame.run_rnd_model(config, trial, env_name)


def run_qrnd_model(config, trial):
    PPO_ProcgenGame.run_qrnd_model(config, trial, env_name)


def run_forward_model(config, trial):
    PPO_ProcgenGame.run_forward_model(config, trial, env_name)


def run_fwd_model(config, trial):
    PPO_ProcgenGame.run_fwd_model(config, trial, env_name)


def run_icm_model(config, trial):
    PPO_ProcgenGame.run_icm_model(config, trial, env_name)


def run_cnd_model(config, trial):
    PPO_ProcgenGame.run_cnd_model(config, trial, env_name)
