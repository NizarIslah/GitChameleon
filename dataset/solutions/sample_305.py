# library: librosa
# version: 0.7.0
# extra_dependencies: ['pip==24.0', 'numba==0.46', 'llvmlite==0.30', 'joblib==0.12', 'numpy==1.16.0', 'audioread==2.1.5', 'scipy==1.1.0', 'resampy==0.2.2', 'soundfile==0.10.2']
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

    # Set the default window length if it is not already specified.
    if win_length is None:
        win_length = frame_length // 2

    # Set the default hop if it is not already specified.
    if hop_length is None:
        hop_length = frame_length // 4

    # Pad the time series so that frames are centered
    if center:
        y = np.pad(y, frame_length // 2, mode=pad_mode)

    # Frame audio.
    y_frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)

    # Calculate minimum and maximum periods
    min_period = max(int(np.floor(sr / fmax)), 1)
    max_period = min(int(np.ceil(sr / fmin)), frame_length - win_length - 1)

    # Calculate cumulative mean normalized difference function.
    # Autocorrelation.
    a = np.fft.rfft(y_frames, frame_length, axis=0)
    b = np.fft.rfft(y_frames[win_length::-1, :], frame_length, axis=0)
    acf_frames = np.fft.irfft(a * b, frame_length, axis=0)[win_length:]
    acf_frames[np.abs(acf_frames) < 1e-6] = 0

    # Energy terms.
    energy_frames = np.cumsum(y_frames**2, axis=0)
    energy_frames = energy_frames[win_length:, :] - energy_frames[:-win_length, :]
    energy_frames[np.abs(energy_frames) < 1e-6] = 0

    # Difference function.
    yin_frames = energy_frames[0, :] + energy_frames - 2 * acf_frames

    # Cumulative mean normalized difference function.
    yin_numerator = yin_frames[min_period : max_period + 1, :]
    tau_range = np.arange(1, max_period + 1)[:, None]
    cumulative_mean = np.cumsum(yin_frames[1 : max_period + 1, :], axis=0) / tau_range
    yin_denominator = cumulative_mean[min_period - 1 : max_period, :]
    yin_frames = yin_numerator / (yin_denominator + librosa.util.tiny(yin_denominator))

    parabolic_shifts = np.zeros_like(yin_frames)
    parabola_a = (yin_frames[:-2, :] + yin_frames[2:, :] - 2 * yin_frames[1:-1, :]) / 2
    parabola_b = (yin_frames[2:, :] - yin_frames[:-2, :]) / 2
    parabolic_shifts[1:-1, :] = -parabola_b / (
        2 * parabola_a + librosa.util.tiny(parabola_a)
    )
    parabolic_shifts[np.abs(parabolic_shifts) > 1] = 0

    # Find local minima.
    is_trough = librosa.util.localmax(-yin_frames, axis=0)
    is_trough[0, :] = yin_frames[0, :] < yin_frames[1, :]

    # Find minima below peak threshold.
    is_threshold_trough = np.logical_and(is_trough, yin_frames < trough_threshold)

    # Absolute threshold.
    # "The solution we propose is to set an absolute threshold and choose the
    # smallest value of tau that gives a minimum of d' deeper than
    # this threshold. If none is found, the global minimum is chosen instead."
    global_min = np.argmin(yin_frames, axis=0)
    yin_period = np.argmax(is_threshold_trough, axis=0)
    no_trough_below_threshold = np.all(~is_threshold_trough, axis=0)
    yin_period[no_trough_below_threshold] = global_min[no_trough_below_threshold]

    # Refine peak by parabolic interpolation.
    yin_period = (
        min_period
        + yin_period
        + parabolic_shifts[yin_period, range(yin_frames.shape[1])]
    )

    # Convert period to fundamental frequency.
    return sr / yin_period
