# Add the parent directory to import sys
import os
import sys
import tempfile
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sample_150
from werkzeug.exceptions import NotFound


class TestSafeJoinFail404(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()

    def test_safe_join_success(self):
        # Test a valid sub path
        base_path = self.temp_dir
        sub_path = "valid_subdir"

        # Create the subdirectory to ensure it exists
        os.makedirs(os.path.join(base_path, sub_path), exist_ok=True)

        # This should succeed
        result = sample_150.safe_join_fail_404(base_path, sub_path)

        # Check that the result is the joined path
        expected_path = os.path.join(base_path, sub_path)
        self.assertEqual(result, expected_path)

    def test_safe_join_fail(self):
        # Test an invalid sub path (trying to access parent directory)
        base_path = self.temp_dir
        sub_path = "../outside_base_dir"

        # This should raise a NotFound exception
        with self.assertRaises(NotFound):
            sample_150.safe_join_fail_404(base_path, sub_path)

    def test_safe_join_with_empty_subpath(self):
        # Test with an empty sub path
        base_path = self.temp_dir
        sub_path = ""

        # This should succeed and return the base path
        result = sample_150.safe_join_fail_404(base_path, sub_path)
        # Normalize both paths to avoid trailing slash issues
        self.assertEqual(os.path.normpath(result), os.path.normpath(base_path))

    def tearDown(self):
        # Clean up the temporary directory
        import shutil


base_path = "/var/www/myapp"
sub_path = "../secret.txt"

try:
    joined = safe_join_fail_404(base_path, sub_path)
except werkzeug.exceptions.NotFound as e:
    assertion_result = True
else:
    assertion_result = False
assert assertion_result

base_path = "/var/www/myapp"
sub_path = "secret.txt"
joined = safe_join_fail_404(base_path, sub_path)
assertion_result = joined == "/var/www/myapp/secret.txt"
assert assertion_result
