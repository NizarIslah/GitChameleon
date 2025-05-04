# library: librosa
# version: 0.8.0
# extra_dependencies: ['pip==24.0', 'numba==0.46', 'llvmlite==0.30', 'joblib==0.14', 'numpy==1.16.0', 'audioread==2.1.5', 'scipy==1.1.0', 'resampy==0.2.2', 'soundfile==0.10.2']
import librosa
import numpy as np
import scipy
from typing import Union, Optional

DTypeLike = Union[np.dtype, type]


def compute_griffinlim_cqt(
    y: np.ndarray,
    sr: int,
    C,
    n_iter: int,
    hop_length: int,
    fmin: int,
    bins_per_octave: int,
    tuning: float,
    filter_scale: 1,
    norm: int,
    sparsity: float,
    window: str,
    scale: bool,
    pad_mode: str,
    res_type: str,
    dtype: DTypeLike,
    length: Optional[int],
    momentum: float,
    init: Optional[str],
) -> np.ndarray:
    rng = np.random.RandomState(seed=0)

    return librosa.griffinlim_cqt(C, sr=sr, bins_per_octave=bins_per_octave, init=init)
