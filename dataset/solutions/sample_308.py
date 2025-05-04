# library: librosa
# version: 0.8.0
# extra_dependencies: ['pip==24.0', 'numba==0.46', 'llvmlite==0.30', 'joblib==0.14', 'numpy==1.16.0', 'audioread==2.1.5', 'scipy==1.1.0', 'resampy==0.2.2', 'soundfile==0.10.2']
import librosa
import numpy as np
import scipy
from typing import Union, Optional, Tuple

DTypeLike = Union[np.dtype, type]


def compute_pyin(
    freq: int,
    sr: int,
    y: int,
    fmin: int,
    fmax: int,
    frame_length: int,
    center: bool,
    pad_mode: str,
    win_length: Optional[int],
    hop_length: Optional[int],
    n_thresholds: int,
    beta_parameters: Tuple[int],
    boltzmann_parameter: int,
    resolution: float,
    max_transition_rate: float,
    switch_prob: float,
    no_trough_prob: float,
    fill_na: DTypeLike,
) -> np.ndarray:
    """
    Calculates the fundamental frequency estimation using probabilistic YIN.

    Parameters:
        freq: The frequency of the fundamental frequency in Hz.
        sr: The sampling rate of the audio signal in Hertz.
        y: The audio signal.
        fmin: The minimum frequency to consider in Hz.
        fmax: The maximum frequency to consider in Hz.
        frame_length: The length of the frame in samples.
        center: If True, the signal y is padded so that frame t is centered at y[t * hop_length].
        pad_mode: Padding mode.
        win_length: Window length.
        hop_length: Hop length.
        n_thresholds: Number of thresholds.
        beta_parameters: Beta parameters.
        boltzmann_parameter: Boltzmann parameter.
        resolution: Resolution.
        max_transition_rate: Maximum transition rate.
        switch_prob: Switch probability.
        no_trough_prob: No trough probability.
        fill_na: Fill NA value.

    Returns:
        Time series of fundamental frequencies in Hertz.
    """

    return librosa.pyin(y, fmin=fmin, fmax=fmax, center=center)[0]
