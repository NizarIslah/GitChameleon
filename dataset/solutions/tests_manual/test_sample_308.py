# Add the parent directory to import sys
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import librosa
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_308 import compute_pyin


freq = 110
sr = 22050
y = librosa.tone(freq, duration=1.0)
fmin = 110
fmax = 880
frame_length = 2048
center = False
pad_mode = "reflect"
win_length = None
hop_length = None
# trough_threshold = 0.1

n_thresholds = 100
beta_parameters = (2, 18)
boltzmann_parameter = 2
resolution = 0.1
max_transition_rate = 35.92
switch_prob = 0.01
no_trough_prob = 0.01
fill_na = np.nan

sol = compute_pyin(
    freq,
    sr,
    y,
    fmin,
    fmax,
    frame_length,
    center,
    pad_mode,
    win_length,
    hop_length,
    n_thresholds,
    beta_parameters,
    boltzmann_parameter,
    resolution,
    max_transition_rate,
    switch_prob,
    no_trough_prob,
    fill_na,
)

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
    probs[global_min] += no_trough_prob * np.sum(beta_probs[:n_thresholds_below_min])

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
test_sol = f0
assert np.allclose(test_sol, sol)
assert np.allclose(np.log2(sol), np.log2(freq), rtol=0, atol=1e-2)
