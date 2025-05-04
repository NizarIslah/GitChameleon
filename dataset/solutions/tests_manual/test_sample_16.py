# Add the parent directory to import sys
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_16 import istft

# Sample rate (samples per second)
fs = 8000
# Duration of the signal in seconds
t = 1
# Time axis for the signal
time = torch.linspace(0, t, steps=int(fs * t))
# Frequency of the sine wave in Hz
frequency = 440
# Generate a sine wave
signal = torch.sin(2 * torch.pi * frequency * time)
n_fft = 1024  # Number of FFT points
hop_length = 256  # Number of samples between successive frames
win_length = 1024  # Window length
# Compute STFT
spectrogram = torch.stft(
    signal,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    window=torch.hann_window(win_length),
    normalized=False,
    return_complex=True,
)
expected_shape = signal.shape
assert expected_shape == istft(spectrogram, signal, n_fft, hop_length, win_length).shape
