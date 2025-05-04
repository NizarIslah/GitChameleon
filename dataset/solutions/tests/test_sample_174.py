# Add the parent directory to import sys
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import sample_174


class TestSaveExponential(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing path operations
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up the temporary directory
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_save_exponential_basic(self):
        """Test basic functionality of save_exponential."""
        # Create a simple 2x2x2 array (2 matrices of size 2x2)
        A = np.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],  # Identity matrix
                [[0.0, 1.0], [1.0, 0.0]],  # Swap matrix
            ]
        )

        base_path = self.temp_dir
        sub_path = "test_folder"

        # Call the function
        joined_path, result = sample_174.save_exponential(A, base_path, sub_path)

        # Check the joined path
        expected_path = os.path.join(base_path, sub_path)
        self.assertEqual(joined_path, expected_path)

        # Check the shape of the result
        self.assertEqual(result.shape, A.shape)

        # Check the exponential of the identity matrix (should remain identity)
        np.testing.assert_allclose(
            result[0], np.array([[np.e, 0], [0, np.e]]), rtol=1e-5
        )

        # Check the exponential of the swap matrix
        expected_exp_swap = np.array(
            [[np.cosh(1), np.sinh(1)], [np.sinh(1), np.cosh(1)]]
        )
        np.testing.assert_allclose(result[1], expected_exp_swap, rtol=1e-5)

    def test_save_exponential_zero_matrix(self):
        """Test save_exponential with zero matrices."""
        # Create a 2x3x3 array of zeros (2 zero matrices of size 3x3)
        A = np.zeros((2, 3, 3))

        base_path = self.temp_dir
        sub_path = "zeros"

        # Call the function
        joined_path, result = sample_174.save_exponential(A, base_path, sub_path)

        # Check the joined path
        expected_path = os.path.join(base_path, sub_path)
        self.assertEqual(joined_path, expected_path)

        # Check the shape of the result
        self.assertEqual(result.shape, A.shape)

        # The exponential of a zero matrix is the identity matrix
        for i in range(A.shape[0]):
            expected = np.eye(A.shape[1])
            np.testing.assert_allclose(result[i], expected, rtol=1e-5)

    def test_path_traversal_protection(self):
        """Test that the function protects against path traversal attacks."""
        A = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        base_path = self.temp_dir

        # Try a path traversal attack
        malicious_path = "../../../etc/passwd"

        # The function should raise a 404 error (NotFound)
        with self.assertRaises(sample_174.error404):
            sample_174.save_exponential(A, base_path, malicious_path)

    def test_large_batch(self):
        """Test with a larger batch of matrices."""
        # Create 10 random 4x4 matrices
        np.random.seed(42)  # For reproducibility
        A = np.random.rand(10, 4, 4)

        base_path = self.temp_dir
        sub_path = "large_batch"

        # Call the function
        joined_path, result = sample_174.save_exponential(A, base_path, sub_path)

        # Check the shape of the result
        self.assertEqual(result.shape, A.shape)

        # Verify a few results manually using scipy's expm
        for i in range(min(3, A.shape[0])):  # Check first 3 matrices
            expected = sample_174.linalg.expm(A[i])
            np.testing.assert_allclose(result[i], expected, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
