# library: librosa
# version: 0.7.0
# extra_dependencies: ['pip==24.0', 'numba==0.46', 'llvmlite==0.30', 'joblib==0.14', 'numpy==1.16.0', 'audioread==2.1.5', 'scipy==1.1.0', 'resampy==0.2.2', 'soundfile==0.10.2']
import librosa
import numpy as np
import scipy
import scipy.optimize
from typing import Union, Optional

DTypeLike = Union[np.dtype, type]


def compute_mel_to_audio(
    y: np.ndarray,
    sr: int,
    S: np.ndarray,
    M: np.ndarray,
    n_fft: int,
    hop_length: Optional[int],
    win_length: Optional[int],
    window: str,
    center: bool,
    pad_mode: str,
    power: float,
    n_iter: int,
    length: Optional[int],
    dtype: DTypeLike,
) -> np.ndarray:
    np.random.seed(seed=0)

    return librosa.feature.inverse.mel_to_audio(M)
