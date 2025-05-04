# library: scipy
# version: 1.8.1
# extra_dependencies: ['numpy==1.21.6']
import numpy as np
from scipy.signal import hilbert


def compute_hilbert_transform(
    a: np.ndarray, b: np.ndarray, dtype=np.float64
) -> np.ndarray:
    # compute_hilbert_transform should return the Hilbert transform of the
    # a and b arrays stacked vertically, with safe casting and the specified
    # dtype.
    # raise TypeError if needed

    if not (
        np.can_cast(a.dtype, dtype, casting="safe")
        and np.can_cast(b.dtype, dtype, casting="safe")
    ):
        raise TypeError("Unsafe casting from input dtype to specified dtype")

    a_cast = a.astype(dtype, copy=False)
    b_cast = b.astype(dtype, copy=False)

    stacked = np.vstack((a_cast, b_cast))

    result = hilbert(stacked)

    if dtype == np.float32:
        complex_dtype = np.complex64
    elif dtype == np.float64:
        complex_dtype = np.complex128
    else:
        complex_dtype = np.complex128

    return result.astype(complex_dtype)
