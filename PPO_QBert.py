import PPO_AtariGame

env_name = 'Qbert-v0'


def test(config, path):
    PPO_AtariGame.test(config, path, env_name)


def run_baseline(config, trial):
    PPO_AtariGame.run_baseline(config, trial, env_name)


def run_forward_model(config, trial):
    PPO_AtariGame.run_forward_model(config, trial, env_name)