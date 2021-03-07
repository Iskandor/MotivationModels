import json


def load_config_file(learning_algorithm):
    filename = None

    if learning_algorithm == 'a2c':
        filename = './config/a2c.config.json'
    if learning_algorithm == 'ddpg':
        filename = './config/ddpg.config.json'
    if learning_algorithm == 'dqn':
        filename = './config/dqn.config.json'
    if learning_algorithm == 'ppo':
        filename = './config/ppo.config.json'

    assert filename is not None, "Unknown learning algorithm {0:s}".format(learning_algorithm)

    with open(filename) as f:
        config = json.load(f)

    return config



