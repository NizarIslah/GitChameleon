import os

# Add the parent directory to the path so we can import the sample module
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_278 import compute_rms


duration = 2.0
frequency = 440
sr = 22050

t = np.linspace(0, duration, int(sr * duration), endpoint=False)

y = 0.5 * np.sin(2 * np.pi * frequency * t)

assert np.array_equal(librosa.feature.rms(y=y), compute_rms(y))
