# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_6 import gamma_ln
from scipy.special import gammaln as scipy_gammaln


class TestGammaLn(unittest.TestCase):
    """Test cases for the gamma_ln function in sample_6.py."""

    def test_basic_positive_values(self):
        """Test gamma_ln with basic positive values."""
        input_tensor = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0], dtype=torch.float32)
        result = gamma_ln(input_tensor)
        expected = torch.from_numpy(scipy_gammaln(input_tensor.numpy()))

        # Check that the result is a tensor
        self.assertIsInstance(result, torch.Tensor)

        # Check that the values match the expected output
        torch.testing.assert_close(result, expected)

        # Check that the dtype is preserved
        self.assertEqual(result.dtype, input_tensor.dtype)

        # For integers n > 0, gamma_ln(n) = ln((n-1)!)
        # So gamma_ln(3) = ln(2!) = ln(2) ≈ 0.693
        self.assertTrue(
            torch.isclose(result[4], torch.tensor(np.log(2.0), dtype=torch.float32))
        )

    def test_negative_non_integer_values(self):
        """Test gamma_ln with negative non-integer values."""
        input_tensor = torch.tensor([-0.5, -1.5, -2.5], dtype=torch.float32)
        result = gamma_ln(input_tensor)
        expected = torch.from_numpy(scipy_gammaln(input_tensor.numpy()))

        torch.testing.assert_close(result, expected)

        # The gamma function has poles at negative integers, but is defined for negative non-integers
        # Check that the values are finite for negative non-integers
        self.assertTrue(torch.all(torch.isfinite(result)))

    def test_integer_values(self):
        """Test gamma_ln with integer values."""
        input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        result = gamma_ln(input_tensor)

        # For integer n, gamma(n) = (n-1)!, so gamma_ln(n) = ln((n-1)!)
        expected_values = [0.0, 0.0, np.log(2.0), np.log(6.0), np.log(24.0)]
        expected = torch.tensor(expected_values, dtype=torch.float32)

        torch.testing.assert_close(result, expected)

        # Check specific values
        self.assertTrue(
            torch.isclose(result[0], torch.tensor(0.0))
        )  # gamma_ln(1) = ln(0!) = ln(1) = 0
        self.assertTrue(
            torch.isclose(result[1], torch.tensor(0.0))
        )  # gamma_ln(2) = ln(1!) = ln(1) = 0

    def test_large_values(self):
        """Test gamma_ln with large values."""
        input_tensor = torch.tensor([10.0, 20.0, 50.0], dtype=torch.float32)
        result = gamma_ln(input_tensor)
        expected = torch.from_numpy(scipy_gammaln(input_tensor.numpy()))

        # For large values, gamma_ln grows approximately like x*ln(x)
        torch.testing.assert_close(result, expected)

        # Check that gamma_ln is increasing for x > 1
        self.assertTrue(result[1] > result[0])
        self.assertTrue(result[2] > result[1])

    def test_small_values(self):
        """Test gamma_ln with small positive values close to zero."""
        input_tensor = torch.tensor([1e-5, 1e-10, 1e-20], dtype=torch.float64)
        result = gamma_ln(input_tensor)
        expected = torch.from_numpy(scipy_gammaln(input_tensor.numpy()))

        # For small positive x, gamma_ln(x) ≈ -ln(x) - gamma*x where gamma is Euler's constant
        torch.testing.assert_close(result, expected)

        # For very small positive x, gamma_ln(x) approaches infinity as x approaches 0
        # The smaller the x, the larger the gamma_ln value
        # Check that the values are finite
        self.assertTrue(torch.all(torch.isfinite(result)))

    def test_multi_dimensional_tensors(self):
        """Test gamma_ln with multi-dimensional tensors."""
        input_tensor = torch.tensor([[0.5, 1.0], [1.5, 2.0]], dtype=torch.float32)
        result = gamma_ln(input_tensor)
        expected = torch.from_numpy(scipy_gammaln(input_tensor.numpy()))

        # Check shape
        self.assertEqual(result.shape, input_tensor.shape)

        # Check values
        torch.testing.assert_close(result, expected)

    def test_non_tensor_input(self):
        """Test gamma_ln with non-tensor input (should raise TypeError)."""
        with self.assertRaises(TypeError):
            # List is not a tensor
            gamma_ln([0.5, 1.0, 1.5])

    def test_compare_with_scipy_implementation(self):
        """Test that gamma_ln matches scipy's gammaln implementation across a range of values."""
        # Create a range of values to test
        input_tensor = torch.linspace(0.1, 10.0, 100, dtype=torch.float64)

        # Calculate with our function
        result = gamma_ln(input_tensor)

        # Calculate with scipy directly
        expected = torch.from_numpy(scipy_gammaln(input_tensor.numpy()))

        # Check that they match closely
        torch.testing.assert_close(result, expected)

        # Additional check for dtype preservation
        self.assertEqual(result.dtype, input_tensor.dtype)

        # Check the recurrence relation: gamma_ln(x+1) = gamma_ln(x) + ln(x)
        x_values = input_tensor[:-1]  # All but the last value
        x_plus_1 = x_values + 1.0

        gamma_ln_x = gamma_ln(x_values)
        gamma_ln_x_plus_1 = gamma_ln(x_plus_1)

        # Check that gamma_ln(x+1) ≈ gamma_ln(x) + ln(x)
        expected_relation = gamma_ln_x + torch.log(x_values)
        torch.testing.assert_close(
            gamma_ln_x_plus_1, expected_relation, rtol=1e-10, atol=1e-10
        )


if __name__ == "__main__":
    unittest.main()
