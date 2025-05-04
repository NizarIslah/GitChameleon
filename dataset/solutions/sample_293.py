# library: librosa
# version: 0.6.0
# extra_dependencies: ['pip==24.0', 'scikit-learn==0.21.0', 'numba==0.46', 'llvmlite==0.30', 'joblib==0.12', 'numpy==1.16.0', 'audioread==2.1.5', 'scipy==1.1.0', 'resampy==0.2.2', 'soundfile==0.10.2']
import librosa
import numpy as np


def compute_times_like(
    y: np.ndarray, sr: int, hop_length: int, D: np.ndarray
) -> np.ndarray:
    """
    Compute the times vector of a spectrogram.

    Parameters:
        y: The audio signal.
        sr: The sampling rate of the audio signal in Hertz.
        hop_length: The number of samples between successive frames.
        D: The spectrogram.

    Returns:
        The computed times vector.
    """

    if np.isscalar(D):
        frames = np.arange(D)  # type: ignore
    else:
        frames = np.arange(D.shape[-1])  # type: ignore
    offset = 0
    samples = (np.asanyarray(frames) * hop_length + offset).astype(int)

    return np.asanyarray(samples) / float(sr)
