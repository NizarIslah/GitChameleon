#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for sample_316.py
"""

import os

# Add the parent directory to the path so we can import the sample module
import sys
import unittest

import librosa
import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset/samples"))
)
from sample_316 import compute_mfcc_to_mel


filename = librosa.util.example_audio_file()
y, sr = librosa.load(filename)
mfcc = librosa.feature.mfcc(y=y, sr=sr)

sol = compute_mfcc_to_mel(mfcc)

np.random.seed(seed=0)
test_sol = librosa.feature.inverse.mfcc_to_mel(mfcc)
assert np.allclose(test_sol, sol)
