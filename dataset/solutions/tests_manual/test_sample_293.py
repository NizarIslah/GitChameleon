#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for sample_293.py
"""

import os

# Add the parent directory to the path so we can import the sample module
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_293 import compute_times_like


filename = librosa.util.example_audio_file()
y, sr = librosa.load(filename)
D = librosa.stft(y)
hop_length = 512

sol = compute_times_like(y, sr, hop_length, D)

if np.isscalar(D):
    frames = np.arange(D)  # type: ignore
else:
    frames = np.arange(D.shape[-1])  # type: ignore
offset = 0
samples = (np.asanyarray(frames) * hop_length + offset).astype(int)

test_sol = np.asanyarray(samples) / float(sr)
assert np.array_equal(test_sol, sol)
