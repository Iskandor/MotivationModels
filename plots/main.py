from plots import plot

if __name__ == '__main__':
    config = [
        {'env': 'aeris_navigate', 'algorithm': 'ppo', 'model': 'baseline', 'id': '1'},
        # {'env': 'aeris_navigate', 'algorithm': 'ppo', 'model': 'rnd', 'id': '2'},
        {'env': 'aeris_navigate', 'algorithm': 'ppo', 'model': 'dop', 'id': '3'},
        # {'env': 'aeris_navigate', 'algorithm': 'ppo', 'model': 'dop', 'id': '4'},
        # {'env': 'aeris_navigate', 'algorithm': 'ppo', 'model': 'dop', 'id': '6'},
        {'env': 'aeris_navigate', 'algorithm': 'ppo', 'model': 'dop', 'id': '7'},
        {'env': 'aeris_navigate', 'algorithm': 'ppo', 'model': 'dop_ref', 'id': '5'},
    ]

    plot(config, plot_details=True, window=10000)

    config = [
        {'env': 'aeris_hazards', 'algorithm': 'ppo', 'model': 'baseline', 'id': '1'},
        {'env': 'aeris_hazards', 'algorithm': 'ppo', 'model': 'dop', 'id': '3'}
    ]

    plot(config, plot_details=True, window=10000)

    config = [
        {'env': 'aeris_fragiles', 'algorithm': 'ppo', 'model': 'baseline', 'id': '1' }
    ]

    plot(config, plot_details=True, window=10000)

    # config = [
    #     {'env': 'mspacman', 'algorithm': 'ppo', 'model': 'baseline', 'id': '1'}
    # ]

    # config = [
    #     {'env': 'aeris_navigate', 'algorithm': 'ppo', 'model': 'baseline', 'id': '6'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ppo', 'model': 'baseline', 'id': '7'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ppo', 'model': 'baseline', 'id': '8'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ppo', 'model': 'baseline', 'id': '9'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ppo', 'model': 'baseline', 'id': '10'},
    # ]

    # plot(config, plot_details=False, window=10000)
