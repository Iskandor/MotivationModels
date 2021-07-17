from plots import plot

if __name__ == '__main__':
    # config = [
    #     {'env': 'aeris_navigate', 'algorithm': 'ppo', 'model': 'baseline', 'id': '1'},
    #     # {'env': 'aeris_navigate', 'algorithm': 'ppo', 'model': 'rnd', 'id': '2'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ppo', 'model': 'dop', 'id': '3'},
    #     # {'env': 'aeris_navigate', 'algorithm': 'ppo', 'model': 'dop', 'id': '4'},
    #     # {'env': 'aeris_navigate', 'algorithm': 'ppo', 'model': 'dop', 'id': '6'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ppo', 'model': 'dop', 'id': '7'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ppo', 'model': 'dop_ref', 'id': '5'},
    # ]
    #
    # config = [
    #     {'env': 'aeris_hazards', 'algorithm': 'ppo', 'model': 'baseline', 'id': '1'},
    #     {'env': 'aeris_hazards', 'algorithm': 'ppo', 'model': 'dop', 'id': '3'}
    # ]
    #
    # config = [
    #     {'env': 'aeris_fragiles', 'algorithm': 'ppo', 'model': 'baseline', 'id': '1' }
    # ]

    # config = [
    #     {'env': 'mspacman', 'algorithm': 'ppo', 'model': 'baseline', 'id': '1'}
    # ]

    # config = [
    #     {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'rnd', 'id': '2'}
    # ]

    config = [
        {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'baseline', 'id': '1' },
        {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'baseline', 'id': '2' },
        {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'rnd', 'id': '3'},
        {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'qrnd', 'id': '4'},
        {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '10'},
        {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '11'},
        {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '12'},
    ]

    # config = [
    #     {'env': 'aeris_hazards', 'algorithm': 'ddpg', 'model': 'baseline', 'id': '1' }
    # ]
    #
    # config = [
    #     {'env': 'aeris_fragiles', 'algorithm': 'ddpg', 'model': 'baseline', 'id': '1' }
    # ]

    plot(config, plot_details=[11,12], window=10000)
