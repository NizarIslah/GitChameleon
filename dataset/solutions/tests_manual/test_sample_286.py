import os

# Add the parent directory to the path so we can import the sample module
import sys
import unittest
from unittest.mock import patch

import librosa
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_286 import compute_griffinlim


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
test_sol = librosa.griffinlim(
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
assert np.allclose(test_sol, sol)
