from plots.aeris_plots import plot_aeris

if __name__ == '__main__':
    config = [{
        'env': 'aeris_navigate',
        'model': 'baseline',
        'id': '1'
    }]

    plot_aeris(config, plot_details=True)

    config = [{
        'env': 'aeris_hazards',
        'model': 'baseline',
        'id': '1'
    }]

    plot_aeris(config, plot_details=True)

    config = [{
        'env': 'aeris_fragiles',
        'model': 'baseline',
        'id': '1'
    }]

    plot_aeris(config, plot_details=True)
