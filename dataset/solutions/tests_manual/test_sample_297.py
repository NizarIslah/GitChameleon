import os

# Add the parent directory to the path so we can import the sample module
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_297 import compute_tone


frequency = 440
sr = 22050
length = sr

sol = compute_tone(frequency, sr, length)
phi = -np.pi * 0.5
test_sol = np.cos(2 * np.pi * frequency * np.arange(length) / sr + phi)
assert np.array_equal(test_sol, sol)
