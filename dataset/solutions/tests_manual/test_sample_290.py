# Test file for sample_290.py
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_290 import compute_fourier_tempogram


filename = librosa.util.example_audio_file()
y, sr = librosa.load(filename)
hop_length = 512
oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

sol = compute_fourier_tempogram(oenv, sr, hop_length)
test_sol = librosa.feature.fourier_tempogram(
    onset_envelope=oenv, sr=sr, hop_length=hop_length
)
assert np.array_equal(test_sol, sol)
