import os

# Add the directory containing sample_282.py to the Python path
import sys
import unittest

import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "samples"))
)

from sample_282 import compute_extraction


duration = 2.0
frequency = 440
sr = 22050
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

y = 0.5 * np.sin(2 * np.pi * frequency * t)
y = y.astype(np.float32)

sol = librosa.feature.melspectrogram(y=y, sr=sr)
M_from_y, float32_bool = compute_extraction(y, sr)
assert np.array_equal(sol, M_from_y)
assert float32_bool
