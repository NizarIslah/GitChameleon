#!/usr/bin/env python3
# Test file for sample_295.py

import os

# Add the parent directory to the path so we can import the sample module
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dataset", "samples"))
from sample_295 import compute_samples_like


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
