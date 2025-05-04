# library: librosa
# version: 0.7.0
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

    if fmin is None:
        fmin = librosa.note_to_hz("C1")

    angles = np.empty(C.shape, dtype=np.complex64)
    if init == "random":
        angles[:] = np.exp(2j * np.pi * rng.rand(*C.shape))
    elif init is None:
        angles[:] = 1.0

    rebuilt = 0.0

    for _ in range(n_iter):
        tprev = rebuilt

        inverse = librosa.constantq.icqt(
            C * angles,
            sr=sr,
            hop_length=hop_length,
            bins_per_octave=bins_per_octave,
            fmin=fmin,
            tuning=tuning,
            filter_scale=filter_scale,
            window=window,
            length=length,
            res_type=res_type,
        )

        rebuilt = librosa.constantq.cqt(
            inverse,
            sr=sr,
            bins_per_octave=bins_per_octave,
            n_bins=C.shape[0],
            hop_length=hop_length,
            fmin=fmin,
            tuning=tuning,
            filter_scale=filter_scale,
            window=window,
            res_type=res_type,
        )

        angles[:] = rebuilt - (momentum / (1 + momentum)) * tprev
        angles[:] /= np.abs(angles) + 1e-16

    return librosa.constantq.icqt(
        C * angles,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        tuning=tuning,
        filter_scale=filter_scale,
        fmin=fmin,
        window=window,
        length=length,
        res_type=res_type,
    )
