# Test file for sample_299.py
import os

# Add the samples directory to the path so we can import the module
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dataset", "samples"))
from sample_299 import compute_chirp


import scipy

fmin = 110
fmax = 110 * 64
duration = 1
sr = 22050
linear = True

sol = compute_chirp(fmin, fmax, duration, sr, linear)
period = 1.0 / sr
phi = -np.pi * 0.5
method = "linear" if linear else "logarithmic"
test_sol = scipy.signal.chirp(
    np.arange(int(duration * sr)) / sr,
    fmin,
    duration,
    fmax,
    method=method,
    phi=phi / np.pi * 180,  # scipy.signal.chirp uses degrees for phase offset
)
assert np.array_equal(test_sol, sol)
