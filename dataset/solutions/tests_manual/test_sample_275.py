import os

# Add the parent directory to the path so we can import the module
import sys
import unittest

import librosa
import numpy as np
from scipy.spatial.distance import cdist

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_275 import compute_dtw


X = np.array([[1, 3, 3, 8, 1]])
Y = np.array([[2, 0, 0, 8, 7, 2]])

gt_D = np.array(
    [
        [1.0, 2.0, 3.0, 10.0, 16.0, 17.0],
        [2.0, 4.0, 5.0, 8.0, 12.0, 13.0],
        [3.0, 5.0, 7.0, 10.0, 12.0, 13.0],
        [9.0, 11.0, 13.0, 7.0, 8.0, 14.0],
        [10, 10.0, 11.0, 14.0, 13.0, 9.0],
    ]
)
assert np.array_equal(gt_D, compute_dtw(X, Y))
