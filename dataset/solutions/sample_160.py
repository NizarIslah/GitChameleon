# library: scipy
# version: 1.8.1
# extra_dependencies: ['numpy==1.21.6']
import numpy as np
from scipy.stats import hmean


def count_unique_hmean(data: np.ndarray) -> int:
    # data shape: (n, m)
    # n: number of arrays
    # m: number of elements in each array
    hmean_values = []
    for arr in data:
        if np.isnan(arr).any():
            hm = np.nan
        else:
            hm = hmean(arr)
        hmean_values.append(hm)

    hmean_values = np.asarray(hmean_values)
    non_nan_vals = hmean_values[~np.isnan(hmean_values)]
    counts_non_nan = np.unique(non_nan_vals).shape[0]
    nan_count = np.sum(np.isnan(hmean_values))
    return counts_non_nan + nan_count
