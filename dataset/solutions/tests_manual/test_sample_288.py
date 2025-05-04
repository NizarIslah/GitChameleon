import os

# Add the parent directory to the path so we can import the sample module
import sys
import unittest

import librosa
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_288 import compute_lpc_coef


filename = librosa.util.example_audio_file()
y, sr = librosa.load(filename)
order = 2

sol = compute_lpc_coef(y, sr, order)
test_sol = librosa.lpc(y, order)
assert np.array_equal(test_sol, sol)
