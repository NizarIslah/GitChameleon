import os
import sys
import unittest
from typing import Optional

import numpy as np

# Make sure we import our solution module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sample_285
from sample_285 import compute_griffinlim


filename = librosa.util.example_audio_file()
y, sr = librosa.load(filename)
momentum = 0.99
S = np.abs(librosa.stft(y))
random_state = 0
rng = np.random.RandomState(seed=random_state)
n_iter = 32
hop_length = None
win_length = None
window = "hann"
center = True
dtype = np.float32
length = None
pad_mode = "reflect"
n_fft = 2 * (S.shape[0] - 1)

rng = np.random.RandomState(seed=random_state)
sol = compute_griffinlim(
    y,
    sr,
    S,
    random_state,
    n_iter,
    hop_length,
    win_length,
    window,
    center,
    dtype,
    length,
    pad_mode,
    n_fft,
)

rng = np.random.RandomState(seed=random_state)
angles = np.exp(2j * np.pi * rng.rand(*S.shape))

rebuilt = 0.0

for _ in range(n_iter):
    tprev = rebuilt

    inverse = istft(
        S * angles,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        dtype=dtype,
        length=length,
    )

    rebuilt = stft(
        inverse,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    angles[:] = rebuilt - (momentum / (1 + momentum)) * tprev
    angles[:] /= np.abs(angles) + 1e-16

test_sol = istft(
    S * angles,
    hop_length=hop_length,
    win_length=win_length,
    window=window,
    center=center,
    dtype=dtype,
    length=length,
)
assert np.allclose(test_sol, sol)
