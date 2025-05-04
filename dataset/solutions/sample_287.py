# library: librosa
# version: 0.6.0
# extra_dependencies: ['pip==24.0', 'numba==0.46', 'llvmlite==0.30', 'joblib==0.12', 'numpy==1.16.0', 'audioread==2.1.5', 'scipy==1.1.0', 'resampy==0.2.2', 'soundfile==0.10.2']
import librosa
import numpy as np


def compute_lpc_coef(y: np.ndarray, sr: int, order: int) -> np.ndarray:
    """
    Compute the Linear Prediction Coefficients of an audio signal.

    Parameters:
        y: The audio signal.
        sr: The sampling rate of the audio signal in Hertz.
        order: Order of the linear filter.

    Returns:
        LP prediction error coefficients, i.e. filter denominator polynomial.
    """

    dtype = y.dtype.type
    ar_coeffs = np.zeros(order + 1, dtype=dtype)
    ar_coeffs[0] = dtype(1)
    ar_coeffs_prev = np.zeros(order + 1, dtype=dtype)
    ar_coeffs_prev[0] = dtype(1)
    fwd_pred_error = y[1:]
    bwd_pred_error = y[:-1]
    den = np.dot(fwd_pred_error, fwd_pred_error) + np.dot(
        bwd_pred_error, bwd_pred_error
    )
    for i in range(order):
        if den <= 0:
            raise FloatingPointError("numerical error, input ill-conditioned?")
        reflect_coeff = dtype(-2) * np.dot(bwd_pred_error, fwd_pred_error) / dtype(den)
        ar_coeffs_prev, ar_coeffs = ar_coeffs, ar_coeffs_prev
        for j in range(1, i + 2):
            ar_coeffs[j] = ar_coeffs_prev[j] + reflect_coeff * ar_coeffs_prev[i - j + 1]
        fwd_pred_error_tmp = fwd_pred_error
        fwd_pred_error = fwd_pred_error + reflect_coeff * bwd_pred_error
        bwd_pred_error = bwd_pred_error + reflect_coeff * fwd_pred_error_tmp
        q = dtype(1) - reflect_coeff**2
        den = q * den - bwd_pred_error[-1] ** 2 - fwd_pred_error[0] ** 2
        fwd_pred_error = fwd_pred_error[1:]
        bwd_pred_error = bwd_pred_error[:-1]
    return ar_coeffs
