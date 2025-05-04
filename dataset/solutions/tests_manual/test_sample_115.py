import os

# Add the parent directory to the path so we can import the module
import sys
import unittest

import numpy as np
from scipy import linalg, sparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_115 import compute_matrix_exponential


A = sparse.lil_matrix((3, 3))
A[0, 0] = 4
A[1, 1] = 5
A[1, 2] = 6
output = compute_matrix_exponential(A)
assertion_value = np.allclose(output.todense(), linalg.expm(A).todense())
assert assertion_value
