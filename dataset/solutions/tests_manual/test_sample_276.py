import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_276 import compute_dtw


class TestComputeDTW(unittest.TestCase):
    def test_compute_dtw_with_valid_metric(self):
        """Test a patched version of compute_dtw with a valid metric."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        Y = np.array([[1, 2, 3], [4, 5, 6]])

        # Create a patched version of the function with a valid metric
        def patched_compute_dtw(X, Y):
            import librosa
            from scipy.spatial.distance import cdist


X = np.array([[1, 3, 3, 8, 1]])
Y = np.array([[2, 0, 0, 8, 7, 2]])

gt_D = np.array([[1., 2., 3., 10., 16., 17.],
 [2., 4., 5., 8., 12., 13.],
 [3., 5., 7., 10., 12., 13.],
 [9., 11., 13., 7., 8., 14.],
 [10, 10., 11., 14., 13., 9.]])
assert np.array_equal(gt_D, compute_dtw(X, Y))