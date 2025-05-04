import os
import sys
import unittest
import numpy as np

# Add the parent directory to the path so we can import the sample module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_279 import compute_fill_diagonal


mut_x = np.ones((8, 12))
radius = 0.25

assert np.array_equal(
    librosa.fill_off_diagonal(mut_x, radius), compute_fill_diagonal(mut_x, radius)
)
