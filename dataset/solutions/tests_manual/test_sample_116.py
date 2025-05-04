import os
import sys
import unittest

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_116 import compute_matrix_exponential


A = sparse.lil_matrix((3, 3))
A[0, 0] = 4
A[1, 1] = 5
A[1, 2] = 6
output = compute_matrix_exponential(A)
assertion_value = np.allclose(output.todense(), sparse.linalg.expm(A).todense())
assert assertion_value
