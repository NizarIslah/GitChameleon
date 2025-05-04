# Add the parent directory to import sys
import os
import sys
import tempfile
import unittest

import numpy as np
from werkzeug.exceptions import NotFound

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_173 import save_exponential

from scipy.linalg import expm  # Used for expected results


class TestSaveExponential(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up the temporary directory
        import shutil


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
