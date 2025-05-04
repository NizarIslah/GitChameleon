# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_300 import compute_chirp


fmin = 110
fmax = 110 * 64
duration = 1
sr = 22050
linear = True

sol = compute_chirp(fmin, fmax, duration, sr, linear)

test_sol = librosa.chirp(fmin=fmin, fmax=fmax, duration=duration, sr=sr)
assert np.array_equal(test_sol, sol)
