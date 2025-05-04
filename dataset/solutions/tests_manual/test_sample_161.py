# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_161 import compute_hilbert_transform


a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
assertion_value = False
try:
    compute_hilbert_transform(a, b, dtype=np.float32)
except TypeError:
    assertion_value = True
assert assertion_value
b = b.astype(np.float32)
computed = compute_hilbert_transform(a, b, dtype=np.float32)
expected = hilbert(np.vstack([a.astype(np.float64), b.astype(np.float64)])).astype(
    dtype=np.complex64
)
assertion_value = np.allclose(computed, expected) & (computed.dtype == np.complex64)
assert assertion_value
a = a.astype(np.float64)
b = b.astype(np.float64)
computed = compute_hilbert_transform(a, b, dtype=np.float64)
expected = expected.astype(dtype=np.complex128)
assertion_value = np.allclose(computed, expected) & (computed.dtype == np.complex128)
assert assertion_value
