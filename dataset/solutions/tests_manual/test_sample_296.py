import os

# Add the directory containing sample_296.py to the Python path
import sys
import unittest

import librosa
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "samples"))
)

# Import the function to test
from sample_296 import compute_samples_like


filename = librosa.util.example_audio_file()
y, sr = librosa.load(filename)
D = librosa.stft(y)
hop_length = 512

sol = compute_samples_like(y, sr, D, hop_length)

if np.isscalar(D):
    frames = np.arange(D)  # type: ignore
else:
    frames = np.arange(D.shape[-1])  # type: ignore
offset = 0
test_sol = (np.asanyarray(frames) * hop_length + offset).astype(int)

assert np.array_equal(test_sol, sol)
