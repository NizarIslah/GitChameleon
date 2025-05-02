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

gt_D = np.array([[1., 2., 3., 10., 16., 17.],
 [2., 4., 5., 8., 12., 13.],
 [3., 5., 7., 10., 12., 13.],
 [9., 11., 13., 7., 8., 14.],
 [10, 10., 11., 14., 13., 9.]])
assert np.array_equal(gt_D, compute_dtw(X, Y))