import torch
import numpy as np


def one_hot_code(values, value_dim):
    code = torch.zeros((values.shape[0], value_dim), dtype=torch.float32, device=values.device)
    code = code.scatter(1, values, 1.0)
    return code


def stratify_sampling(x, n_samples, stratify):
    """Perform stratify sampling of a tensor.

    parameters
    ----------
    x: np.ndarray or torch.Tensor
        Array to sample from. Sampels from first dimension.

    n_samples: int
        Number of samples to sample

    stratify: tuple of int
        Size of each subgroup. Note that the sum of all the sizes
        need to be equal to `x.shape[']`.
    """
    n_total = x.shape[0]
    assert sum(stratify) == n_total

    n_strat_samples = [int(i * n_samples / n_total) for i in stratify]
    cum_n_samples = np.cumsum([0] + stratify)
    sampled_idcs = []
    for i, n_strat_sample in enumerate(n_strat_samples):
        sampled_idcs.append(np.random.choice(range(cum_n_samples[i], cum_n_samples[i + 1]),
                                             replace=False,
                                             size=n_strat_sample))

    # might not be correct number of samples due to rounding
    n_current_samples = sum(n_strat_samples)
    if n_current_samples < n_samples:
        delta_n_samples = n_samples - n_current_samples
        # might actually resample same as before, but it's only for a few
        sampled_idcs.append(np.random.choice(range(n_total), replace=False, size=delta_n_samples))

    samples = x[np.concatenate(sampled_idcs), ...]

    return samples
