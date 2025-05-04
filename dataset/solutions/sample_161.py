# library: scipy
# version: 1.11.1
# extra_dependencies: ['numpy==1.25.1']
import numpy as np
from scipy.signal import hilbert


def compute_hilbert_transform(a, b, dtype=np.float64):
    # compute_hilbert_transform should return the Hilbert transform of the
    # a and b arrays stacked vertically, with safe casting and the specified
    # dtype.
    # raise TypeError if needed

    stacked = np.vstack((a, b), dtype=dtype, casting="safe")
    return hilbert(stacked)
