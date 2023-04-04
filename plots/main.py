from plots import plot
from plots.analytic_table import compute_table_values


def atari_env(plots=True, tables=True):
    key = 're'
    config = [
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'baseline', 'id': '1', 'legend': 'Baseline'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'rnd', 'id': '2', 'legend': 'RND'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'icm', 'id': '30', 'legend': 'ICM'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '49', 'legend': 'SND-V', 'mode': 'mch'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '42', 'legend': 'SND-STD'},
        {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'cnd', 'id': '44', 'legend': 'SND-VIC'},
        # {'env': 'montezuma', 'algorithm': 'ppo', 'model': 'fwd', 'id': '45', 'legend': 'SP'},
    ]

    if tables:
        compute_table_values(config, keys=[key])

    if plots:
        plot('montezuma', config, labels=['external reward', 'score'], keys=[key], plot_details=[], window=10000)
    #
    #
    config = [
        {'env': 'gravitar', 'algorithm': 'ppo', 'model': 'baseline', 'id': '1', 'legend': 'Baseline'},
        {'env': 'gravitar', 'algorithm': 'ppo', 'model': 'rnd', 'id': '2', 'legend': 'RND'},
        # {'env': 'gravitar', 'algorithm': 'ppo', 'model': 'icm', 'id': '10', 'legend': 'ICM'},
        # {'env': 'gravitar', 'algorithm': 'ppo', 'model': 'cnd', 'id': '14', 'legend': 'SND-V', 'mode': 'mch'},
        {'env': 'gravitar', 'algorithm': 'ppo', 'model': 'cnd', 'id': '11', 'legend': 'SND-STD'},
        {'env': 'gravitar', 'algorithm': 'ppo', 'model': 'cnd', 'id': '13', 'legend': 'SND-VIC'},
        # {'env': 'gravitar', 'algorithm': 'ppo', 'model': 'fwd', 'id': '12', 'legend': 'SP'},
    ]

    if tables:
        compute_table_values(config, keys=[key])

    if plots:
        plot('gravitar', config, labels=['external reward', 'score'], keys=[key], plot_details=[], window=10000)

    config = [
        {'env': 'venture', 'algorithm': 'ppo', 'model': 'baseline', 'id': '1', 'legend': 'Baseline'},
        {'env': 'venture', 'algorithm': 'ppo', 'model': 'rnd', 'id': '2', 'legend': 'RND'},
        # {'env': 'venture', 'algorithm': 'ppo', 'model': 'icm', 'id': '6', 'legend': 'ICM'},
        # {'env': 'venture', 'algorithm': 'ppo', 'model': 'cnd', 'id': '10', 'legend': 'SND-V', 'mode': 'mch'},
        {'env': 'venture', 'algorithm': 'ppo', 'model': 'cnd', 'id': '4', 'legend': 'SND-STD'},
        {'env': 'venture', 'algorithm': 'ppo', 'model': 'cnd', 'id': '8', 'legend': 'SND-VIC'},
        # {'env': 'venture', 'algorithm': 'ppo', 'model': 'fwd', 'id': '7', 'legend': 'SP'},
    ]

    if tables:
        compute_table_values(config, keys=[key])

    if plots:
        plot('venture', config, labels=['external reward', 'score'], keys=[key], plot_details=[], window=10000)

    config = [
        {'env': 'pitfall', 'algorithm': 'ppo', 'model': 'baseline', 'id': '1', 'legend': 'Baseline'},
        {'env': 'pitfall', 'algorithm': 'ppo', 'model': 'rnd', 'id': '2', 'legend': 'RND'},
        # {'env': 'pitfall', 'algorithm': 'ppo', 'model': 'icm', 'id': '5', 'legend': 'ICM'},
        # {'env': 'pitfall', 'algorithm': 'ppo', 'model': 'baseline', 'id': '1', 'legend': 'SND-V'},
        {'env': 'pitfall', 'algorithm': 'ppo', 'model': 'cnd', 'id': '4', 'legend': 'SND-STD'},
        {'env': 'pitfall', 'algorithm': 'ppo', 'model': 'cnd', 'id': '7', 'legend': 'SND-VIC'},
        # {'env': 'pitfall', 'algorithm': 'ppo', 'model': 'fwd', 'id': '6', 'legend': 'SP'},
    ]

    if tables:
        compute_table_values(config, keys=[key])

    if plots:
        plot('pitfall', config, labels=['external reward', 'score'], keys=[key], plot_details=[], window=10000)

    config = [
        {'env': 'private_eye', 'algorithm': 'ppo', 'model': 'baseline', 'id': '1', 'legend': 'Baseline'},
        {'env': 'private_eye', 'algorithm': 'ppo', 'model': 'rnd', 'id': '2', 'legend': 'RND'},
        # {'env': 'private_eye', 'algorithm': 'ppo', 'model': 'icm', 'id': '5', 'legend': 'ICM'},
        # {'env': 'private_eye', 'algorithm': 'ppo', 'model': 'cnd', 'id': '8', 'legend': 'SND-V', 'mode': 'mch'},
        {'env': 'private_eye', 'algorithm': 'ppo', 'model': 'cnd', 'id': '4', 'legend': 'SND-STD'},
        {'env': 'private_eye', 'algorithm': 'ppo', 'model': 'cnd', 'id': '7', 'legend': 'SND-VIC'},
        # {'env': 'private_eye', 'algorithm': 'ppo', 'model': 'fwd', 'id': '6', 'legend': 'SP'},
    ]

    if tables:
        compute_table_values(config, keys=[key])

    if plots:
        plot('private_eye', config, labels=['external reward', 'score'], keys=[key], plot_details=[7], window=10000)

    config = [
        {'env': 'solaris', 'algorithm': 'ppo', 'model': 'baseline', 'id': '1', 'legend': 'Baseline'},
        {'env': 'solaris', 'algorithm': 'ppo', 'model': 'rnd', 'id': '2', 'legend': 'RND'},
        # {'env': 'solaris', 'algorithm': 'ppo', 'model': 'icm', 'id': '5', 'legend': 'ICM'},
        # {'env': 'solaris', 'algorithm': 'ppo', 'model': 'cnd', 'id': '8', 'legend': 'SND-V'},
        {'env': 'solaris', 'algorithm': 'ppo', 'model': 'cnd', 'id': '4', 'legend': 'SND-STD'},
        {'env': 'solaris', 'algorithm': 'ppo', 'model': 'cnd', 'id': '7', 'legend': 'SND-VIC'},
        # {'env': 'solaris', 'algorithm': 'ppo', 'model': 'fwd', 'id': '6', 'legend': 'SP'},
    ]

    if tables:
        compute_table_values(config, keys=[key])

    if plots:
        plot('solaris', config, labels=['external reward', 'score'], keys=[key], plot_details=[], window=10000)


def procgen_env(plots=True, tables=True):
    key = 're'

    config = [
        {'env': 'caveflyer', 'algorithm': 'ppo', 'model': 'baseline', 'id': '1', 'legend': 'Baseline'},
        {'env': 'caveflyer', 'algorithm': 'ppo', 'model': 'rnd', 'id': '2', 'legend': 'RND'},
        # {'env': 'caveflyer', 'algorithm': 'ppo', 'model': 'icm', 'id': '6', 'legend': 'ICM'},
        # {'env': 'caveflyer', 'algorithm': 'ppo', 'model': 'cnd', 'id': '8', 'mode': 'mch', 'legend': 'SND-V'},
        {'env': 'caveflyer', 'algorithm': 'ppo', 'model': 'cnd', 'id': '4', 'legend': 'SND-STD'},
        {'env': 'caveflyer', 'algorithm': 'ppo', 'model': 'cnd', 'id': '7', 'legend': 'SND-VIC'},
        # {'env': 'caveflyer', 'algorithm': 'ppo', 'model': 'fwd', 'id': '5', 'legend': 'SP'},
        {'env': 'caveflyer', 'algorithm': 'ppo', 'model': 'cnd', 'id': '9', 'legend': 'SND-VINV'},
    ]

    if tables:
        compute_table_values(config, keys=[key])
    if plots:
        plot('caveflyer', config, labels=['external reward', 'score'], keys=['re'], plot_details=[], window=100000)


    config = [
        {'env': 'coinrun', 'algorithm': 'ppo', 'model': 'baseline', 'id': '1', 'legend': 'Baseline'},
        {'env': 'coinrun', 'algorithm': 'ppo', 'model': 'rnd', 'id': '2', 'legend': 'RND'},
        # {'env': 'coinrun', 'algorithm': 'ppo', 'model': 'icm', 'id': '6', 'legend': 'ICM'},
        # {'env': 'coinrun', 'algorithm': 'ppo', 'model': 'cnd', 'id': '8', 'mode': 'mch', 'legend': 'SND-V'},
        {'env': 'coinrun', 'algorithm': 'ppo', 'model': 'cnd', 'id': '4', 'legend': 'SND-STD'},
        {'env': 'coinrun', 'algorithm': 'ppo', 'model': 'cnd', 'id': '7', 'legend': 'SND-VIC'},
        # {'env': 'coinrun', 'algorithm': 'ppo', 'model': 'fwd', 'id': '5', 'legend': 'SP'},
        {'env': 'coinrun', 'algorithm': 'ppo', 'model': 'cnd', 'id': '9', 'legend': 'SND-VINV'},
    ]

    if tables:
        compute_table_values(config, keys=[key])
    if plots:
        plot('coinrun', config, labels=['external reward', 'score'], keys=['re'], plot_details=[], window=100000)

    config = [
        {'env': 'climber', 'algorithm': 'ppo', 'model': 'baseline', 'id': '1', 'legend': 'Baseline'},
        {'env': 'climber', 'algorithm': 'ppo', 'model': 'rnd', 'id': '2', 'legend': 'RND'},
        # {'env': 'climber', 'algorithm': 'ppo', 'model': 'icm', 'id': '6', 'legend': 'ICM'},
        # {'env': 'climber', 'algorithm': 'ppo', 'model': 'cnd', 'id': '8', 'mode': 'mch', 'legend': 'SND-V'},
        {'env': 'climber', 'algorithm': 'ppo', 'model': 'cnd', 'id': '4', 'legend': 'SND-STD'},
        {'env': 'climber', 'algorithm': 'ppo', 'model': 'cnd', 'id': '7', 'legend': 'SND-VIC'},
        # {'env': 'climber', 'algorithm': 'ppo', 'model': 'fwd', 'id': '5', 'legend': 'SP'},
        {'env': 'climber', 'algorithm': 'ppo', 'model': 'cnd', 'id': '9', 'legend': 'SND-VINV'},
    ]

    if tables:
        compute_table_values(config, keys=[key])
    if plots:
        plot('climber', config, labels=['external reward', 'score'], keys=['re'], plot_details=[], window=100000)

    config = [
        {'env': 'jumper', 'algorithm': 'ppo', 'model': 'baseline', 'id': '1', 'legend': 'Baseline', 'mode': 'mch'},
        {'env': 'jumper', 'algorithm': 'ppo', 'model': 'rnd', 'id': '2', 'legend': 'RND'},
        # {'env': 'jumper', 'algorithm': 'ppo', 'model': 'icm', 'id': '6', 'legend': 'ICM'},
        # {'env': 'jumper', 'algorithm': 'ppo', 'model': 'cnd', 'id': '8', 'mode': 'mch', 'legend': 'SND-V'},
        {'env': 'jumper', 'algorithm': 'ppo', 'model': 'cnd', 'id': '4', 'legend': 'SND-STD'},
        {'env': 'jumper', 'algorithm': 'ppo', 'model': 'cnd', 'id': '7', 'legend': 'SND-VIC'},
        # {'env': 'jumper', 'algorithm': 'ppo', 'model': 'fwd', 'id': '5', 'legend': 'SP'},
        {'env': 'jumper', 'algorithm': 'ppo', 'model': 'cnd', 'id': '9', 'legend': 'SND-VINV'},
    ]

    if tables:
        compute_table_values(config, keys=[key])
    if plots:
        plot('jumper', config, labels=['external reward', 'score'], keys=['re'], plot_details=[], window=100000)


if __name__ == '__main__':
    atari_env(plots=True)
    procgen_env(plots=True)
