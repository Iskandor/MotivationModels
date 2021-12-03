import PPO_HardAtariGame

env_name = 'MontezumaRevengeNoFrameskip-v4'


def test(config, path):
    PPO_HardAtariGame.test(config, path, env_name)


def run_baseline(config, trial):
    PPO_HardAtariGame.run_baseline(config, trial, env_name)


def run_rnd_model(config, trial):
    PPO_HardAtariGame.run_rnd_model(config, trial, env_name)


def run_forward_model(config, trial):
    PPO_HardAtariGame.run_forward_model(config, trial, env_name)
