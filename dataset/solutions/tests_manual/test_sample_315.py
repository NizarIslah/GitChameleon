#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for sample_315.py which contains the compute_mfcc_to_mel function.
"""

import os

# Add the parent directory to the path so we can import the sample module
import sys
import unittest

import librosa
import numpy as np
import scipy

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset/samples"))
)
from sample_315 import compute_mfcc_to_mel

filename = librosa.util.example_audio_file()
y, sr = librosa.load(filename)
mfcc = librosa.feature.mfcc(y=y, sr=sr)

sol = compute_mfcc_to_mel(mfcc)


def mfcc_to_mel(mfcc, n_mels=128, dct_type=2, norm="ortho", ref=1.0):
    logmel = scipy.fftpack.idct(mfcc, axis=0, type=dct_type, norm=norm, n=n_mels)
    return librosa.db_to_power(logmel, ref=ref)


np.random.seed(seed=0)
test_sol = mfcc_to_mel(mfcc)
assert np.allclose(test_sol, sol)
