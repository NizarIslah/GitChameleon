# library: librosa
# version: 0.7.0
# extra_dependencies: ['pip==24.0', 'numba==0.46', 'llvmlite==0.30', 'joblib==0.14', 'numpy==1.16.0', 'audioread==2.1.5', 'scipy==1.1.0', 'resampy==0.2.2', 'soundfile==0.10.2']
import librosa
import numpy as np
import scipy
from typing import Union, Optional, Tuple

DTypeLike = Union[np.dtype, type]


def compute_pyin(
    freq: int,
    sr: int,
    y: np.ndarray,
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

    if win_length is None:
        win_length = frame_length // 2

    if hop_length is None:
        hop_length = frame_length // 4

    if center:
        y = np.pad(y, frame_length // 2, mode=pad_mode)

    y_frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)

    min_period = max(int(np.floor(sr / fmax)), 1)
    max_period = min(int(np.ceil(sr / fmin)), frame_length - win_length - 1)

    a = np.fft.rfft(y_frames, frame_length, axis=0)
    b = np.fft.rfft(y_frames[win_length::-1, :], frame_length, axis=0)
    acf_frames = np.fft.irfft(a * b, frame_length, axis=0)[win_length:]
    acf_frames[np.abs(acf_frames) < 1e-6] = 0

    energy_frames = np.cumsum(y_frames**2, axis=0)
    energy_frames = energy_frames[win_length:, :] - energy_frames[:-win_length, :]
    energy_frames[np.abs(energy_frames) < 1e-6] = 0

    yin_frames = energy_frames[0, :] + energy_frames - 2 * acf_frames

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

    thresholds = np.linspace(0, 1, n_thresholds + 1)
    beta_cdf = scipy.stats.beta.cdf(thresholds, beta_parameters[0], beta_parameters[1])
    beta_probs = np.diff(beta_cdf)

    yin_probs = np.zeros_like(yin_frames)
    for i, yin_frame in enumerate(yin_frames.T):
        is_trough = librosa.util.localmax(-yin_frame, axis=0)
        is_trough[0] = yin_frame[0] < yin_frame[1]
        (trough_index,) = np.nonzero(is_trough)

        if len(trough_index) == 0:
            continue

        trough_heights = yin_frame[trough_index]
        trough_thresholds = trough_heights[:, None] < thresholds[None, 1:]

        trough_positions = np.cumsum(trough_thresholds, axis=0) - 1
        n_troughs = np.count_nonzero(trough_thresholds, axis=0)
        trough_prior = scipy.stats.boltzmann.pmf(
            trough_positions, boltzmann_parameter, n_troughs
        )
        trough_prior[~trough_thresholds] = 0

        probs = np.sum(trough_prior * beta_probs, axis=1)
        global_min = np.argmin(trough_heights)
        n_thresholds_below_min = np.count_nonzero(~trough_thresholds[global_min, :])
        probs[global_min] += no_trough_prob * np.sum(
            beta_probs[:n_thresholds_below_min]
        )

        yin_probs[trough_index, i] = probs

    yin_period, frame_index = np.nonzero(yin_probs)

    period_candidates = min_period + yin_period
    period_candidates = period_candidates + parabolic_shifts[yin_period, frame_index]
    f0_candidates = sr / period_candidates

    n_bins_per_semitone = int(np.ceil(1.0 / resolution))
    n_pitch_bins = int(np.floor(12 * n_bins_per_semitone * np.log2(fmax / fmin))) + 1

    max_semitones_per_frame = round(max_transition_rate * 12 * hop_length / sr)
    transition_width = max_semitones_per_frame * n_bins_per_semitone + 1

    transition = librosa.sequence.transition_local(
        n_pitch_bins, transition_width, window="triangle", wrap=False
    )

    transition = np.block(
        [
            [(1 - switch_prob) * transition, switch_prob * transition],
            [switch_prob * transition, (1 - switch_prob) * transition],
        ]
    )

    bin_index = 12 * n_bins_per_semitone * np.log2(f0_candidates / fmin)
    bin_index = np.clip(np.round(bin_index), 0, n_pitch_bins).astype(int)

    observation_probs = np.zeros((2 * n_pitch_bins, yin_frames.shape[1]))
    observation_probs[bin_index, frame_index] = yin_probs[yin_period, frame_index]
    voiced_prob = np.clip(np.sum(observation_probs[:n_pitch_bins, :], axis=0), 0, 1)
    observation_probs[n_pitch_bins:, :] = (1 - voiced_prob[None, :]) / n_pitch_bins

    p_init = np.zeros(2 * n_pitch_bins)
    p_init[n_pitch_bins:] = 1 / n_pitch_bins

    states = librosa.sequence.viterbi(observation_probs, transition, p_init=p_init)

    freqs = fmin * 2 ** (np.arange(n_pitch_bins) / (12 * n_bins_per_semitone))
    f0 = freqs[states % n_pitch_bins]
    voiced_flag = states < n_pitch_bins
    if fill_na is not None:
        f0[~voiced_flag] = fill_na

    return f0
