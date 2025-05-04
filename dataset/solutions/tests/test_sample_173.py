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

        shutil.rmtree(self.temp_dir)

    def test_valid_path_join(self):
        """Test that valid paths are joined correctly."""
        # Create a simple 2x2 matrix
        A = np.array([[[1.0, 2.0], [3.0, 4.0]]])

        # Test with a valid sub path
        sub_path = "valid_folder"
        joined_path, result = save_exponential(A, self.temp_dir, sub_path)

        # Check that the path was joined correctly
        expected_path = os.path.join(self.temp_dir, sub_path)
        self.assertEqual(joined_path, expected_path)

        # Compute expected result using scipy.linalg.expm
        expected_result = np.array([expm(A[0])])
        np.testing.assert_allclose(result, expected_result, rtol=1e-6)

    def test_path_traversal_attempt(self):
        """Test that path traversal attempts raise a 404 error."""
        A = np.array([[[1.0, 0.0], [0.0, 1.0]]])

        # Try a path traversal attack
        with self.assertRaises(NotFound):
            save_exponential(A, self.temp_dir, "../../../etc/passwd")

    def test_multiple_matrices(self):
        """Test with multiple matrices in the batch."""
        # Create a batch of 3 matrices
        A = np.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],  # Identity matrix
                [[0.0, 1.0], [1.0, 0.0]],  # Pauli X matrix
                [[0.0, -1j], [1j, 0.0]],  # Pauli Y matrix
            ]
        )

        joined_path, result = save_exponential(A, self.temp_dir, "batch")

        # Check the path
        expected_path = os.path.join(self.temp_dir, "batch")
        self.assertEqual(joined_path, expected_path)

        # Check the results
        # For identity matrix, exp(I) = e^I = e * I
        np.testing.assert_allclose(
            result[0], np.array([[np.e, 0], [0, np.e]]), rtol=1e-8
        )

        # For Pauli X, exp(X) has a known form
        np.testing.assert_allclose(
            result[1],
            np.array([[np.cosh(1), np.sinh(1)], [np.sinh(1), np.cosh(1)]]),
            rtol=1e-8,
        )

        # For Pauli Y, exp(Y) has a known form
        np.testing.assert_allclose(
            result[2],
            np.array([[np.cosh(1), -1j * np.sinh(1)], [1j * np.sinh(1), np.cosh(1)]]),
            rtol=1e-8,
        )

    def test_empty_subpath(self):
        """Test with an empty sub path."""
        A = np.array([[[0.0, 0.0], [0.0, 0.0]]])

        joined_path, result = save_exponential(A, self.temp_dir, "")

        # Check the path (normalize both to avoid trailing slash issues)
        self.assertEqual(os.path.normpath(joined_path), os.path.normpath(self.temp_dir))

        # For zero matrix, exp(0) = I
        np.testing.assert_allclose(
            result, np.array([[[1.0, 0.0], [0.0, 1.0]]]), rtol=1e-8
        )

    def test_none_subpath(self):
        """Test with None as sub path should raise TypeError."""
        A = np.array([[[0.0, 0.0], [0.0, 0.0]]])

        with self.assertRaises(TypeError):
            save_exponential(A, self.temp_dir, None)


if __name__ == "__main__":
    unittest.main()
