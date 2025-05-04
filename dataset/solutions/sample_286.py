# library: librosa
# version: 0.7.0
# extra_dependencies: ['pip==24.0', 'numba==0.46', 'llvmlite==0.30', 'joblib==0.12', 'numpy==1.16.0', 'audioread==2.1.5', 'scipy==1.1.0', 'resampy==0.2.2', 'soundfile==0.10.2']
import librosa
import numpy as np
from librosa import istft, stft
from typing import Union, Optional

DTypeLike = Union[np.dtype, type]


def compute_griffinlim(
    y: np.ndarray,
    sr: int,
    S: np.ndarray,
    random_state: int,
    n_iter: int,
    hop_length: Optional[int],
    win_length: Optional[int],
    window: str,
    center: bool,
    dtype: DTypeLike,
    length: Optional[int],
    pad_mode: str,
    n_fft: int,
) -> np.ndarray:
    """
    Compute waveform from a linear scale magnitude spectrogram using the Griffin-Lim transformation.

    Parameters:
        y: Audio timeseries.
        sr: Sampling rate.
        S: short-time Fourier transform magnitude matrix.
        random_state: Random state for the random number generator.
        n_iter: Number of iterations.
        hop_length: Hop length.
        win_length: Window length.
        window: Window function.
        center: If True, the signal y is padded so that frame t is centered at y[t * hop_length]. If False, then frame t begins at y[t * hop_length].
        dtype: Data type of the output.
        length: Length of the output signal.
        pad_mode: Padding mode.
        n_fft: FFT size.

    Returns:
        The Griffin-Lim waveform.
    """
    rng = np.random.RandomState(seed=random_state)

    return librosa.griffinlim(
        S,
        n_iter,
        hop_length,
        win_length,
        window,
        center,
        dtype,
        length,
        pad_mode,
        momentum,
        random_state,
    )
