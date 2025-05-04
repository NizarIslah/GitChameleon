# library: librosa
# version: 0.7.0
# extra_dependencies: ['pip==24.0', 'numba==0.46', 'llvmlite==0.30', 'joblib==0.14', 'numpy==1.16.0', 'audioread==2.1.5', 'scipy==1.1.0', 'resampy==0.2.2', 'soundfile==0.10.2']
import librosa
import numpy as np
import scipy


def compute_mfcc_to_mel(
    mfcc: np.ndarray,
    n_mels: int = 128,
    dct_type: int = 2,
    norm: str = "ortho",
    ref: float = 1.0,
) -> np.ndarray:
    """
    Invert Mel-frequency cepstral coefficients to approximate a Mel power spectrogram.

    Parameters:
        mfcc (np.ndarray): Mel-frequency cepstral coefficients.
        n_mels (int): Number of Mel bands to generate.
        dct_type (int): Type of DCT to use.
        norm (str): Normalization to use.
        ref: Reference power for (inverse) decibel calculation

    Returns:
        An approximate Mel power spectrum recovered from mfcc.
    """
    np.random.seed(seed=0)

    return librosa.feature.inverse.mfcc_to_mel(mfcc)
