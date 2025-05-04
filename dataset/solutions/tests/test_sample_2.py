# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_2 import erf
from scipy.special import erf as scipy_erf


class TestErf(unittest.TestCase):
    """Test cases for the erf function in sample_2.py."""

    def test_basic_positive_values(self):
        """Test erf with basic positive values."""
        input_tensor = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0], dtype=torch.float32)
        result = erf(input_tensor)
        expected = torch.from_numpy(scipy_erf(input_tensor.numpy()))

        # Check that the result is a tensor
        self.assertIsInstance(result, torch.Tensor)

        # Check that the values match the expected output
        torch.testing.assert_close(result, expected)

        # Check that the dtype is preserved
        self.assertEqual(result.dtype, input_tensor.dtype)

    def test_negative_values(self):
        """Test erf with negative values."""
        input_tensor = torch.tensor([-0.5, -1.0, -1.5, -2.0], dtype=torch.float32)
        result = erf(input_tensor)
        expected = torch.from_numpy(scipy_erf(input_tensor.numpy()))

        torch.testing.assert_close(result, expected)

        # erf is an odd function, so erf(-x) = -erf(x)
        positive_input = torch.abs(input_tensor)
        positive_result = erf(positive_input)
        self.assertTrue(torch.allclose(-positive_result, result))

    def test_zero_values(self):
        """Test erf with zero values."""
        input_tensor = torch.tensor([0.0], dtype=torch.float32)
        result = erf(input_tensor)

        # erf(0) should be 0
        self.assertTrue(torch.allclose(result, torch.tensor([0.0])))

    def test_large_values(self):
        """Test erf with large positive and negative values."""
        input_tensor = torch.tensor([10.0, -10.0, 20.0, -20.0], dtype=torch.float32)
        result = erf(input_tensor)
        expected = torch.from_numpy(scipy_erf(input_tensor.numpy()))

        # For very large values, erf approaches 1 or -1
        torch.testing.assert_close(result, expected)

        # Check that large positive values approach 1
        self.assertTrue(torch.allclose(result[0], torch.tensor(1.0), atol=1e-7))
        # Check that large negative values approach -1
        self.assertTrue(torch.allclose(result[1], torch.tensor(-1.0), atol=1e-7))

    def test_small_values(self):
        """Test erf with small values close to zero."""
        input_tensor = torch.tensor([1e-10, -1e-10, 1e-5, -1e-5], dtype=torch.float64)
        result = erf(input_tensor)
        expected = torch.from_numpy(scipy_erf(input_tensor.numpy()))

        # For very small values, erf(x) ≈ (2/√π) * x
        torch.testing.assert_close(result, expected)

        # For small x, erf(x) ≈ (2/√π) * x ≈ 1.128379 * x
        approx_factor = 2 / np.sqrt(np.pi)
        approx_result = input_tensor * approx_factor
        self.assertTrue(torch.allclose(result, approx_result, atol=1e-10))

    def test_multi_dimensional_tensors(self):
        """Test erf with multi-dimensional tensors."""
        input_tensor = torch.tensor([[0.0, 0.5], [1.0, 1.5]], dtype=torch.float32)
        result = erf(input_tensor)
        expected = torch.from_numpy(scipy_erf(input_tensor.numpy()))

        # Check shape
        self.assertEqual(result.shape, input_tensor.shape)

        # Check values
        torch.testing.assert_close(result, expected)

    def test_non_tensor_input(self):
        """Test erf with non-tensor input (should raise TypeError)."""
        with self.assertRaises(AttributeError):
            # List doesn't have numpy() method
            erf([0.0, 0.5, 1.0])

    def test_compare_with_scipy_implementation(self):
        """Test that erf matches scipy's erf implementation across a range of values."""
        # Create a range of values to test
        input_tensor = torch.linspace(-5.0, 5.0, 100, dtype=torch.float64)

        # Calculate with our function
        result = erf(input_tensor)

        # Calculate with scipy directly
        expected = torch.from_numpy(scipy_erf(input_tensor.numpy()))

        # Check that they match closely
        torch.testing.assert_close(result, expected)

        # Additional check for dtype preservation
        self.assertEqual(result.dtype, input_tensor.dtype)


if __name__ == "__main__":
    unittest.main()
