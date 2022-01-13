from plots import plot

if __name__ == '__main__':
    # config = [
    #     {'env': 'aeris_navigate', 'algorithm': 'ppo', 'model': 'baseline', 'id': '1'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ppo', 'model': 'baseline', 'id': '8'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ppo', 'model': 'baseline', 'id': '9'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ppo', 'model': 'baseline', 'id': '10'},
    # ]
    #
    # plot('aeris_navigate_baseline', config, plot_details=[1], window=10000)

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

    # config = [
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'baseline', 'id': '1'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'baseline', 'id': '2'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'rnd', 'id': '3'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'qrnd', 'id': '4'},
    #     # {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '6'},
    #     # {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '7'},
    #     # {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '8'},
    #     # {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '9'},
    #     # {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '10'},
    #     # {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '11'},
    #     # {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '12'},
    #     # {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '13'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '14'},
    #     # {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '15'},
    #     # {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '16'},
    #     # {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '17'},
    #     # {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '18'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop_ref', 'id': '19'},
    #     # {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '20'},
    #     # {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '21'},
    #     # {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '23'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '24'},
    # ]
    #
    # plot('aeris_navigate_final', config, plot_details=[], window=10000)

    # config = [
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '6'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '7'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '8'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '9'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '10'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '11'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '12'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '13'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '14'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '15'}
    # ]
    #
    # plot('aeris_navigate_dop_trajectory', config, plot_details=[], window=10000)
    #
    # config = [
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '14'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '16'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '17'},
    # ]
    #
    # plot('aeris_navigate_dop_batch', config, plot_details=[], window=10000)

    # config = [
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '14'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '20'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '21'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '22'},
    # ]
    #
    # plot('aeris_navigate_arbiter', config, plot_details=[22], window=10000)

    # config = [
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'baseline', 'id': '1'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'baseline', 'id': '2'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'rnd', 'id': '3'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'rnd', 'id': '25'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'rnd', 'id': '31'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'rnd', 'id': '32'}
    # ]
    #
    # plot('aeris_navigate_rnd', config, plot_details=[], window=10000)

    # config = [
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'baseline', 'id': '1'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'baseline', 'id': '2'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '27'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '28'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '29'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '30'}
    # ]
    #
    # plot('aeris_navigate_dop', config, plot_details=[], window=10000)

    # config = [
    #     {'env': 'aeris_hazards', 'algorithm': 'ddpg', 'model': 'baseline', 'id': '1'},
    #     {'env': 'aeris_hazards', 'algorithm': 'ddpg', 'model': 'baseline', 'id': '2'},
    #     {'env': 'aeris_hazards', 'algorithm': 'ddpg', 'model': 'dop', 'id': '5'},
    #     {'env': 'aeris_hazards', 'algorithm': 'ddpg', 'model': 'dop_ref', 'id': '6'},
    #     {'env': 'aeris_hazards', 'algorithm': 'ddpg', 'model': 'dop_2', 'id': '7'},
    #     {'env': 'aeris_hazards', 'algorithm': 'ddpg', 'model': 'dop', 'id': '8'},
    #     {'env': 'aeris_hazards', 'algorithm': 'ddpg', 'model': 'dop', 'id': '9'},
    #     {'env': 'aeris_hazards', 'algorithm': 'ddpg', 'model': 'dop', 'id': '10'},
    # ]
    #
    # plot('aeris_hazards_final', config, plot_details=[10], window=10000)

    # config = [
    #     {'env': 'aeris_fragiles', 'algorithm': 'ddpg', 'model': 'baseline', 'id': '1'},
    #     {'env': 'aeris_fragiles', 'algorithm': 'ddpg', 'model': 'baseline', 'id': '2'},
    #     {'env': 'aeris_fragiles', 'algorithm': 'ddpg', 'model': 'dop', 'id': '5'}
    # ]
    #
    # plot('aeris_fragiles_final', config, plot_details=[5], window=10000)

    # config = [
    #     {'env': 'aeris_gather', 'algorithm': 'ddpg', 'model': 'baseline', 'id': '1'},
    #     {'env': 'aeris_gather', 'algorithm': 'ddpg', 'model': 'baseline', 'id': '2'},
    #     {'env': 'aeris_gather', 'algorithm': 'ddpg', 'model': 'dop', 'id': '5'},
    #     {'env': 'aeris_gather', 'algorithm': 'ddpg', 'model': 'dop_ref', 'id': '6'},
    #     {'env': 'aeris_gather', 'algorithm': 'ddpg', 'model': 'dop', 'id': '7'},
    # ]
    #
    # plot('aeris_gather_final', config, plot_details=[5, 7], window=10000)

    # config = [
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'baseline', 'id': '1'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'baseline', 'id': '2'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '41'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '42'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '43'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '44'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '45'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop_2', 'id': '46'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop_2', 'id': '47'},
    # ]
    #
    # plot('aeris_navigate_dop_2heads', config, plot_details=[47,48], window=10000)

    # config = [
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'baseline', 'id': '1'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'baseline', 'id': '2'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop_ref', 'id': '19'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '48'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '49'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '50'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '51'},
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '52'},
    # ]
    #
    # plot('aeris_navigate_dop_reg_term', config, plot_details=[49, 51], window=10000)

    # config = [
    #     {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'baseline', 'id': '1'},
    #    {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'baseline', 'id': '2'},
    #    {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop_ref', 'id': '19'},
    #    {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '55'},
    #    {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '56'},
    #    {'env': 'aeris_navigate', 'algorithm': 'ddpg', 'model': 'dop', 'id': '57'},
    #]

    config = [
        {'env': 'gravitar', 'algorithm': 'ppo', 'model': 'rnd', 'id': '1'}
    ]

    plot('gravitar_final', config, plot_details=[1], window=100000, average_per_step=True, has_score=True)
