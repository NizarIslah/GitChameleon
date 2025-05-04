import os

# Add the parent directory to the path so we can import the module
import sys
import unittest

import numpy as np
import sympy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_176 import custom_computeDFT


class TestCustomDFT(unittest.TestCase):
    def test_dft_size(self):
        """Test that the DFT matrix has the correct size."""
        for n in [1, 2, 4, 8]:
            dft_matrix = custom_computeDFT(n)
            self.assertEqual(dft_matrix.shape, (n, n))

    def test_dft_properties(self):
        """Test that the DFT matrix has the expected mathematical properties."""
        # Test for n=2
        dft_2 = custom_computeDFT(2)
        expected_2 = sympy.Matrix([[1, 1], [1, -1]]) / sympy.sqrt(2)
        self.assertTrue(dft_2.equals(expected_2))

        # Test for n=4
        dft_4 = custom_computeDFT(4)
        # Check that it's unitary (U* × U = I)
        # Convert to numpy for easier computation
        dft_4_np = np.array(dft_4).astype(complex)
        identity = np.eye(4)
        product = np.conjugate(dft_4_np.T) @ dft_4_np
        np.testing.assert_almost_equal(product, identity, decimal=10)

    def test_dft_values(self):
        """Test specific values in the DFT matrix."""
        # For n=4, check some specific values
        dft_4 = custom_computeDFT(4)

        # The first row and column should all be 1/sqrt(n)
        n = 4
        expected_val = 1 / sympy.sqrt(n)

        for i in range(n):
            self.assertEqual(dft_4[0, i], expected_val)
            self.assertEqual(dft_4[i, 0], expected_val)

    def test_dft_idempotent(self):
        """Test that applying DFT four times returns the original matrix."""
        # For n=4, DFT^4 should be the identity matrix
        n = 4
        dft = custom_computeDFT(n)

        # Convert to numpy for easier computation
        dft_np = np.array(dft).astype(complex)

        # Apply DFT four times
        result = dft_np
        for _ in range(3):  # Already have one application
            result = result @ dft_np

        # Should approximately equal the identity matrix
        identity = np.eye(n)
        np.testing.assert_almost_equal(result, identity, decimal=10)


if __name__ == "__main__":
    unittest.main()
