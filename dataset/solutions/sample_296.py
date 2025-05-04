# library: librosa
# version: 0.7.0
# extra_dependencies: ['pip==24.0', 'numba==0.46', 'llvmlite==0.30', 'joblib==0.12', 'numpy==1.16.0', 'audioread==2.1.5', 'scipy==1.1.0', 'resampy==0.2.2', 'soundfile==0.10.2']
import librosa
import numpy as np


def compute_samples_like(
    y: np.ndarray, sr: int, D: np.ndarray, hop_length: int
) -> np.ndarray:
    """
    Compute the samples vector of a spectrogram.

    Parameters:
        y: The audio signal.
        sr: The sampling rate of the audio signal in Hertz.
        D: The spectrogram.

    Returns:
        The computed samples vector.
    """

    return librosa.samples_like(D)
