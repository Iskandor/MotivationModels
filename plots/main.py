from plots import plot

if __name__ == '__main__':
    config = [
        {'env': 'aeris_navigate', 'model': 'baseline', 'id': '1'},
        {'env': 'aeris_navigate', 'model': 'rnd', 'id': '2'},
        {'env': 'aeris_navigate', 'model': 'dop', 'id': '3'},
        {'env': 'aeris_navigate', 'model': 'dop_ref', 'id': '5'},
    ]

    # config = [
    #     {'env': 'aeris_hazards', 'model': 'baseline', 'id': '1'},
    #     {'env': 'aeris_hazards', 'model': 'dop', 'id': '3'}
    # ]

    # config = [
    #     {'env': 'aeris_fragiles', 'model': 'baseline', 'id': '1' }
    # ]

    # config = [
    #     {'env': 'mspacman', 'model': 'baseline', 'id': '1'}
    # ]

    # config = [
    #     {'env': 'aeris_navigate', 'model': 'baseline', 'id': '6'},
    #     {'env': 'aeris_navigate', 'model': 'baseline', 'id': '7'},
    #     {'env': 'aeris_navigate', 'model': 'baseline', 'id': '8'},
    #     {'env': 'aeris_navigate', 'model': 'baseline', 'id': '9'},
    #     {'env': 'aeris_navigate', 'model': 'baseline', 'id': '10'},
    # ]

    plot(config, plot_details=False, window=10000)
