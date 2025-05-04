# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_1 import gamma_ln
from scipy.special import gammaln as scipy_gammaln


class TestGammaLn(unittest.TestCase):
    """Test cases for the gamma_ln function in sample_1.py."""

    def test_basic_positive_values(self):
        """Test gamma_ln with basic positive values."""
        input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gamma_ln(input_tensor)
        expected = torch.from_numpy(scipy_gammaln(input_tensor.numpy()))

        # Check that the result is a tensor
        self.assertIsInstance(result, torch.Tensor)

        # Check that the values match the expected output
        torch.testing.assert_close(result, expected)

    def test_large_positive_values(self):
        """Test gamma_ln with large positive values."""
        input_tensor = torch.tensor([10.0, 20.0, 50.0, 100.0])
        result = gamma_ln(input_tensor)
        expected = torch.from_numpy(scipy_gammaln(input_tensor.numpy()))

        torch.testing.assert_close(result, expected)

    def test_small_positive_values(self):
        """Test gamma_ln with small positive values."""
        input_tensor = torch.tensor([0.1, 0.01, 0.001, 0.5])
        result = gamma_ln(input_tensor)
        expected = torch.from_numpy(scipy_gammaln(input_tensor.numpy()))

        torch.testing.assert_close(result, expected)

    def test_zero_values(self):
        """Test gamma_ln with zero values (should return inf)."""
        input_tensor = torch.tensor([0.0])
        result = gamma_ln(input_tensor)

        # gammaln(0) should be infinity
        self.assertTrue(torch.isinf(result[0]))
        self.assertTrue(result[0] > 0)  # Positive infinity

    def test_negative_values(self):
        """Test gamma_ln with negative values."""
        # Negative integers and half-integers should give infinity or NaN
        input_tensor = torch.tensor([-1.0, -2.0, -3.0, -0.5, -1.5])
        result = gamma_ln(input_tensor)

        # Check that the result matches scipy's implementation
        expected = torch.from_numpy(scipy_gammaln(input_tensor.numpy()))
        torch.testing.assert_close(result, expected)

        # Specifically, negative integers should give infinity
        self.assertTrue(torch.isinf(result[0]))
        self.assertTrue(torch.isinf(result[1]))
        self.assertTrue(torch.isinf(result[2]))

    def test_multi_dimensional_tensors(self):
        """Test gamma_ln with multi-dimensional tensors."""
        input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = gamma_ln(input_tensor)
        expected = torch.from_numpy(scipy_gammaln(input_tensor.numpy()))

        # Check shape
        self.assertEqual(result.shape, input_tensor.shape)

        # Check values
        torch.testing.assert_close(result, expected)

    def test_non_tensor_input(self):
        """Test gamma_ln with non-tensor input (should raise TypeError)."""
        with self.assertRaises(AttributeError):
            # List doesn't have numpy() method
            gamma_ln([1.0, 2.0, 3.0])

    def test_compare_with_scipy_implementation(self):
        """Test that gamma_ln matches scipy's gammaln implementation."""
        # Create a range of values to test
        input_tensor = torch.linspace(0.1, 10.0, 100)

        # Calculate with our function
        result = gamma_ln(input_tensor)

        # Calculate with scipy directly
        expected = torch.from_numpy(scipy_gammaln(input_tensor.numpy()))

        # Check that they match closely
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
