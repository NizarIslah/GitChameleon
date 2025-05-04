# library: librosa
# version: 0.6.0
# extra_dependencies: ['pip==24.0', 'scikit-learn==0.21.0', 'numba==0.46', 'llvmlite==0.30', 'joblib==0.12', 'numpy==1.16.0', 'audioread==2.1.5', 'scipy==1.1.0', 'resampy==0.2.2', 'soundfile==0.10.2']
import librosa
import numpy as np


def compute_tone(frequency: int, sr: int, length: int) -> np.ndarray:
    """
    Constructs a pure tone (cosine) signal at a given frequency.

    Parameters:
        frequency: The frequency of the tone in Hz.
        sr: The sampling rate of the signal in Hz.
        length: The length of the signal in samples.

    Returns:
        np.ndarray: The pure tone signal.
    """

    phi = -np.pi * 0.5
    return np.cos(2 * np.pi * frequency * np.arange(length) / sr + phi)
