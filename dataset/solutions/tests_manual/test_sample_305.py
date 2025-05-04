import os

# Add the parent directory to the path so we can import the sample
import sys
import unittest

import librosa
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_305 import compute_yin

sr = 22050
fmin = 440
fmax = 880
duration = 5.0
period = 1.0 / sr
phi = -np.pi * 0.5
method = "linear"
y = scipy.signal.chirp(
    np.arange(int(duration * sr)) / sr,
    fmin,
    duration,
    fmax,
    method=method,
    phi=phi / np.pi * 180,  # scipy.signal.chirp uses degrees for phase offset
)
frame_length = 2048
center = True
pad_mode = "reflect"
win_length = None
hop_length = None
trough_threshold = 0.1

sol = compute_yin(
    sr,
    fmin,
    fmax,
    duration,
    period,
    phi,
    method,
    y,
    frame_length,
    center,
    pad_mode,
    win_length,
    hop_length,
    trough_threshold,
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

is_trough = librosa.util.localmax(-yin_frames, axis=0)
is_trough[0, :] = yin_frames[0, :] < yin_frames[1, :]

is_threshold_trough = np.logical_and(is_trough, yin_frames < trough_threshold)

global_min = np.argmin(yin_frames, axis=0)
yin_period = np.argmax(is_threshold_trough, axis=0)
no_trough_below_threshold = np.all(~is_threshold_trough, axis=0)
yin_period[no_trough_below_threshold] = global_min[no_trough_below_threshold]

yin_period = (
    min_period + yin_period + parabolic_shifts[yin_period, range(yin_frames.shape[1])]
)

test_sol = sr / yin_period
assert np.allclose(test_sol, sol)
