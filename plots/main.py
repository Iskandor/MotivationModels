from plots import plot
from plots.analytic_table import compute_table_values

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    # config = [
    #     {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'rnd', 'id': '2'},
    #     # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '15'},
    #     # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '16'},
    #     # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '17'},
    #     # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '19'},
    #     {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '20'},
    #     {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '21'},
    #     # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '26'},
    # ]

    # config = [
    #     {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '26'},
    #     {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '27'},
    #     {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '28'},
    #     {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '31'},
    #     {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '32'},
    #     {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '33'},
    #     {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '34'},
    #     {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '35'},
    #     {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '36'},
    # ]
    #
    # compute_table_values(config, keys=['re'])
    # plot('montezuma_bt', config, keys=['re', 'score', 'ri'], plot_details=[], window=10000)

    config = [
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'rnd', 'id': '2'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '38'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '40'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '41'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '42'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '43'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '44'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '46'},
    ]

    compute_table_values(config, keys=['re'])
    plot('montezuma_stdim', config, keys=['re', 'score', 'ri'], plot_details=[43], window=10000)
