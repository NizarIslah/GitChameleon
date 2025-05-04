# library: librosa
# version: 0.8.0
# extra_dependencies: ['pip==24.0', 'numba==0.46', 'llvmlite==0.30', 'numpy==1.16.0', 'audioread==2.1.5', 'scipy==1.1.0', 'resampy==0.2.2', 'soundfile==0.10.2']
import librosa
import numpy as np
import scipy
from typing import Optional


def compute_yin(
    sr: int,
    fmin: int,
    fmax: int,
    duration: float,
    period: float,
    phi: float,
    method: str,
    y: np.ndarray,
    frame_length: int,
    center: bool,
    pad_mode: str,
    win_length: Optional[int],
    hop_length: Optional[int],
    trough_threshold: float,
) -> np.ndarray:
    """
    Calculates the fundamental frequency (F0) estimation using the YIN algorithm.

    Parameters:
        sr: The sampling rate of the audio signal in Hertz.
        fmin: The minimum frequency to consider in Hz.
        fmax: The maximum frequency to consider in Hz.
        duration: The duration of the audio signal in seconds.
        period: The period of the fundamental frequency in seconds.
        phi: The phase of the fundamental frequency in radians.
        method: Interpolation method.
        y: The audio signal.
        frame_length: The length of the frame in samples.
        center: If True, the signal y is padded so that frame t is centered at y[t * hop_length].
        pad_mode: Padding mode.
        win_length: Window length.
        hop_length: Hop length.
        trough_threshold: Absolute threshold for peak estimation.

    Returns:
        The estimated fundamental frequency in Hz.
    """

    return librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr)
