import os

# Add the parent directory to the path so we can import the module
import sys
import unittest

import numpy as np
from scipy import linalg

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_112 import compute_matrix_exponential


A = np.array(
    [
        [
            [0.25264461, 0.67582554, 0.90718149, 0.65460219],
            [0.58271792, 0.4600052, 0.22265374, 0.98210688],
            [0.92575218, 0.66167048, 0.81779481, 0.15405207],
            [0.00820708, 0.7702345, 0.4285001, 0.87567275],
        ],
        [
            [0.48362533, 0.10258182, 0.58965127, 0.89320413],
            [0.11275151, 0.95192602, 0.58950113, 0.78663422],
            [0.64955361, 0.47670695, 0.96824964, 0.74915994],
            [0.71266875, 0.27280891, 0.1771122, 0.45839236],
        ],
        [
            [0.96116073, 0.11138203, 0.59254915, 0.92860822],
            [0.78721405, 0.09705598, 0.88774379, 0.81623277],
            [0.64821764, 0.62400451, 0.53916194, 0.96522881],
            [0.68958095, 0.86514529, 0.41583035, 0.84209827],
        ],
    ]
)
output = compute_matrix_exponential(A)
assertion_value = np.allclose(
    output, np.stack([linalg.expm(A[i]) for i in range(A.shape[0])], axis=0)
)
assert assertion_value
