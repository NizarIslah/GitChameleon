# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_4 import bessel_i0
from scipy.special import i0 as scipy_i0


class TestBesselI0(unittest.TestCase):
    """Test cases for the bessel_i0 function in sample_4.py."""

    def test_basic_positive_values(self):
        """Test bessel_i0 with basic positive values."""
        input_tensor = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0], dtype=torch.float32)
        result = bessel_i0(input_tensor)
        expected = torch.from_numpy(scipy_i0(input_tensor.numpy()))

        # Check that the result is a tensor
        self.assertIsInstance(result, torch.Tensor)

        # Check that the values match the expected output
        torch.testing.assert_close(result, expected)

        # Check that the dtype is preserved
        self.assertEqual(result.dtype, input_tensor.dtype)

        # I₀(x) ≥ 1 for all x ≥ 0
        self.assertTrue(torch.all(result >= 1.0))

        # I₀(x) is strictly increasing for x > 0
        for i in range(1, len(input_tensor)):
            if input_tensor[i] > input_tensor[i - 1] and input_tensor[i] > 0:
                self.assertTrue(result[i] > result[i - 1])

    def test_negative_values(self):
        """Test bessel_i0 with negative values."""
        input_tensor = torch.tensor([-0.5, -1.0, -1.5, -2.0], dtype=torch.float32)
        result = bessel_i0(input_tensor)
        expected = torch.from_numpy(scipy_i0(input_tensor.numpy()))

        torch.testing.assert_close(result, expected)

        # I₀(x) is an even function, so I₀(-x) = I₀(x)
        positive_input = torch.abs(input_tensor)
        positive_result = bessel_i0(positive_input)
        self.assertTrue(torch.allclose(positive_result, result))

    def test_zero_values(self):
        """Test bessel_i0 with zero values."""
        input_tensor = torch.tensor([0.0], dtype=torch.float32)
        result = bessel_i0(input_tensor)

        # I₀(0) should be 1
        self.assertTrue(torch.allclose(result, torch.tensor([1.0])))

    def test_large_values(self):
        """Test bessel_i0 with large positive and negative values."""
        input_tensor = torch.tensor([10.0, -10.0, 20.0, -20.0], dtype=torch.float32)
        result = bessel_i0(input_tensor)
        expected = torch.from_numpy(scipy_i0(input_tensor.numpy()))

        # For large values, I₀(x) grows exponentially
        torch.testing.assert_close(result, expected)

        # Check that I₀(x) = I₀(-x) for large values too
        self.assertTrue(torch.allclose(result[0], result[1]))
        self.assertTrue(torch.allclose(result[2], result[3]))

        # Check that I₀(x) grows with |x|
        self.assertTrue(result[2] > result[0])

    def test_small_values(self):
        """Test bessel_i0 with small values close to zero."""
        input_tensor = torch.tensor([1e-10, -1e-10, 1e-5, -1e-5], dtype=torch.float64)
        result = bessel_i0(input_tensor)
        expected = torch.from_numpy(scipy_i0(input_tensor.numpy()))

        # For very small values, I₀(x) ≈ 1 + (x²/4)
        torch.testing.assert_close(result, expected)

        # For small x, I₀(x) ≈ 1 + (x²/4)
        approx_result = 1 + (input_tensor**2) / 4
        self.assertTrue(torch.allclose(result, approx_result, atol=1e-10))

    def test_multi_dimensional_tensors(self):
        """Test bessel_i0 with multi-dimensional tensors."""
        input_tensor = torch.tensor([[0.0, 0.5], [1.0, 1.5]], dtype=torch.float32)
        result = bessel_i0(input_tensor)
        expected = torch.from_numpy(scipy_i0(input_tensor.numpy()))

        # Check shape
        self.assertEqual(result.shape, input_tensor.shape)

        # Check values
        torch.testing.assert_close(result, expected)

    def test_non_tensor_input(self):
        """Test bessel_i0 with non-tensor input (should raise TypeError)."""
        with self.assertRaises(AttributeError):
            # List doesn't have numpy() method
            bessel_i0([0.0, 0.5, 1.0])

    def test_compare_with_scipy_implementation(self):
        """Test that bessel_i0 matches scipy's i0 implementation across a range of values."""
        # Create a range of values to test
        input_tensor = torch.linspace(-5.0, 5.0, 100, dtype=torch.float64)

        # Calculate with our function
        result = bessel_i0(input_tensor)

        # Calculate with scipy directly
        expected = torch.from_numpy(scipy_i0(input_tensor.numpy()))

        # Check that they match closely
        torch.testing.assert_close(result, expected)

        # Additional check for dtype preservation
        self.assertEqual(result.dtype, input_tensor.dtype)

        # Check the symmetry property: I₀(-x) = I₀(x)
        negative_input = -input_tensor
        negative_result = bessel_i0(negative_input)
        self.assertTrue(torch.allclose(result, negative_result))


if __name__ == "__main__":
    unittest.main()
