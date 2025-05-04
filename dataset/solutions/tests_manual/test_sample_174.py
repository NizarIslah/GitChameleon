# Add the parent directory to import sys
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import sample_174


base_path = "/var/www/myapp"
sub_path = "../secret.txt"
import numpy as np

a = np.random.random((4, 3, 3))
expected = np.zeros(a.shape)
for i in range(expected.shape[0]):
    expected[i] = linalg.expm(a[i])

try:
    joined, results = save_exponential(a, base_path, sub_path)
except werkzeug.exceptions.NotFound as e:
    assertion_result = True
else:
    assertion_result = False
assert assertion_result

base_path = "/var/www/myapp"
sub_path = "secret.txt"

joined, results = save_exponential(a, base_path, sub_path)
assertion_result = joined == "/var/www/myapp/secret.txt" and np.allclose(
    results, expected
)
assert assertion_result
