# library: librosa
# version: 0.7.0
# extra_dependencies: ['pip==24.0', 'numba==0.46', 'llvmlite==0.30', 'joblib==0.12', 'numpy==1.16.0', 'audioread==2.1.5', 'scipy==1.1.0', 'resampy==0.2.2', 'soundfile==0.10.2']
import librosa
import numpy as np


def compute_chirp(
    fmin: int, fmax: int, duration: int, sr: int, linear: bool
) -> np.ndarray:
    """
    Constructs a “chirp” or “sine-sweep” signal. The chirp sweeps from frequency fmin to fmax (in Hz).

    Parameters:
        fmin: The minimum frequency of the chirp in Hz.
        fmax: The maximum frequency of the chirp in Hz.
        duration: The duration of the chirp in seconds.
        sr: The sampling rate of the signal in Hz.

    Returns:
        np.ndarray: The chirp signal.
    """

    return librosa.chirp(fmin=fmin, fmax=fmax, duration=duration, sr=sr)
