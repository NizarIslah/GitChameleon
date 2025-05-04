# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_9 import bessel_i0
from scipy.special import i0 as scipy_i0


class TestBesselI0(unittest.TestCase):
    """Test cases for the bessel_i0 function in sample_9.py."""

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

        # I₀(0) = 1
        self.assertTrue(
            torch.isclose(result[0], torch.tensor(1.0, dtype=torch.float32))
        )

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

    def test_zero_value(self):
        """Test bessel_i0 with zero value."""
        input_tensor = torch.tensor([0.0], dtype=torch.float32)
        result = bessel_i0(input_tensor)

        # I₀(0) should be 1
        self.assertTrue(
            torch.allclose(result, torch.tensor([1.0], dtype=torch.float32))
        )

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

        # For very small values, I₀(x) ≈ 1
        torch.testing.assert_close(result, expected)

        # For small x, I₀(x) ≈ 1 + (x²/4)
        small_x = input_tensor[0]  # 1e-10
        expected_approx = 1.0 + (small_x**2 / 4.0)
        self.assertTrue(
            torch.isclose(result[0], expected_approx, rtol=1e-10, atol=1e-10)
        )

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
        with self.assertRaises(TypeError):
            # List is not a tensor
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

        # Check the even function property: I₀(-x) = I₀(x)
        negative_input = -input_tensor
        negative_result = bessel_i0(negative_input)
        self.assertTrue(torch.allclose(result, negative_result))

        # Check that I₀(x) ≥ 1 for all x
        self.assertTrue(torch.all(result >= 1.0))

    def test_bessel_i0_properties(self):
        """Test mathematical properties of the bessel_i0 function."""
        # I₀(x) is an even function: I₀(-x) = I₀(x)
        x_values = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
        neg_x_values = -x_values

        i0_x = bessel_i0(x_values)
        i0_neg_x = bessel_i0(neg_x_values)

        torch.testing.assert_close(i0_neg_x, i0_x)

        # I₀(0) = 1
        self.assertTrue(
            torch.isclose(
                bessel_i0(torch.tensor([0.0], dtype=torch.float64))[0],
                torch.tensor(1.0, dtype=torch.float64),
            )
        )

        # I₀(x) is strictly increasing for x > 0
        x_increasing = torch.linspace(0.0, 5.0, 100, dtype=torch.float64)
        i0_x_increasing = bessel_i0(x_increasing)

        for i in range(1, len(x_increasing)):
            self.assertTrue(i0_x_increasing[i] > i0_x_increasing[i - 1])

        # I₀(x) is convex for all x
        # We can check this by verifying that the second derivative is positive
        h = 1e-4
        x_values = torch.linspace(-3.0, 3.0, 50, dtype=torch.float64)

        # Compute second derivative using central differences
        i0_x = bessel_i0(x_values)
        i0_x_plus_h = bessel_i0(x_values + h)
        i0_x_minus_h = bessel_i0(x_values - h)

        second_derivative = (i0_x_plus_h - 2 * i0_x + i0_x_minus_h) / (h * h)

        # Check that the second derivative is positive
        self.assertTrue(torch.all(second_derivative > 0))


if __name__ == "__main__":
    unittest.main()
