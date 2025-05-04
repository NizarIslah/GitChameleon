# library: librosa
# version: 0.6.0
# extra_dependencies: ['pip==24.0', 'scikit-learn==0.21.0', 'numba==0.46', 'llvmlite==0.30', 'joblib==0.12', 'numpy==1.16.0', 'audioread==2.1.5', 'scipy==1.1.0', 'resampy==0.2.2', 'soundfile==0.10.2']
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

    if np.isscalar(D):
        frames = np.arange(D)  # type: ignore
    else:
        frames = np.arange(D.shape[-1])  # type: ignore
    offset = 0
    return (np.asanyarray(frames) * hop_length + offset).astype(int)
