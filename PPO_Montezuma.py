import PPO_AtariGame

env_name = 'MontezumaRevengeNoFrameskip-v4'


def test(config, path):
    PPO_AtariGame.test(config, path, env_name)


def run_baseline(config, trial):
    PPO_AtariGame.run_baseline(config, trial, env_name)


def run_rnd_model(config, trial):
    PPO_AtariGame.run_rnd_model(config, trial, env_name)


def run_forward_model(config, trial):
    PPO_AtariGame.run_forward_model(config, trial, env_name)
