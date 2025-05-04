# Test file for sample_298.py
import os

# Add the samples directory to the path so we can import the module
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dataset", "samples"))
from sample_298 import compute_tone


frequency = 440
sr = 22050
length = sr

sol = compute_tone(frequency, sr, length)
test_sol = librosa.tone(frequency, sr=sr, length=length)
assert np.array_equal(test_sol, sol)
