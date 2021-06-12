import numpy as np


def run_basic_analysis(data):
    print("Mean: {0}".format(data['re'].mean(1).mean(0)))
    print("Std.dev: {0}".format(data['re'].std(1).mean(0)))
    print("95%: {0}".format(np.percentile(data['re'], 95, axis=1).mean(0)))


def run_motivation_mixture_analysis(data, n_blocks):
    size = data['fmr'].shape[1]
    block = size // n_blocks
    fmr = data['fmr']
    fme = data['fme']
    mcr = data['mcr']
    mce = data['mce']

    mask_r = mcr > fmr
    result_e = fme > mce
    e_source = []
    for i in range(result_e.shape[0]):
        row = np.extract(mask_r[i], result_e[i])
        e_source.append(np.count_nonzero(row) / len(row))
    print("Surprise source: {0:.2f}%".format(np.array(e_source).mean() * 100))

    for i in range(n_blocks):
        mcr_block = mcr[:, i * block:(i + 1) * block]
        fmr_block = fmr[:, i * block:(i + 1) * block]

        result_mc = fmr_block < mcr_block
        result_fm = fmr_block > mcr_block

        quarter_mc = []
        quarter_fm = []
        for row in result_mc:
            quarter_mc.append(np.count_nonzero(row) / block)
        for row in result_fm:
            quarter_fm.append(np.count_nonzero(row) / block)

        mcr_magnitude = np.append(np.extract(result_mc, mcr_block), [0])
        fmr_magnitued = np.append(np.extract(result_fm, fmr_block), [0])

        print("Q{0} surprise density {1:.2f}% with avg. magnitued {2:.5f} predictive error density {3:.2f}% with avg. magnitued {4:.5f}".format(i+1, np.array(quarter_mc).mean() * 100, mcr_magnitude.mean(), np.array(quarter_fm).mean() * 100, fmr_magnitued.mean()))
