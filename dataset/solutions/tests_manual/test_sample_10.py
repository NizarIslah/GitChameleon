# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_10 import bessel_i1
from scipy.special import i1 as scipy_i1


class TestBesselI1(unittest.TestCase):
    """Test cases for the bessel_i1 function in sample_10.py."""

    def test_basic_positive_values(self):
        """Test bessel_i1 with basic positive values."""
        input_tensor = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0], dtype=torch.float32)
        result = bessel_i1(input_tensor)
        expected = torch.from_numpy(scipy_i1(input_tensor.numpy()))

        # Check that the result is a tensor
        self.assertIsInstance(result, torch.Tensor)

        # Check that the values match the expected output
        torch.testing.assert_close(result, expected)

        # Check that the dtype is preserved
        self.assertEqual(result.dtype, input_tensor.dtype)

        # I₁(0) = 0
        self.assertTrue(
            torch.isclose(result[0], torch.tensor(0.0, dtype=torch.float32))
        )

        # I₁(x) ≥ 0 for all x ≥ 0
        self.assertTrue(torch.all(result >= 0.0))

        # I₁(x) is strictly increasing for x > 0
        for i in range(1, len(input_tensor)):
            if input_tensor[i] > input_tensor[i - 1] and input_tensor[i] > 0:
                self.assertTrue(result[i] > result[i - 1])

    def test_negative_values(self):
        """Test bessel_i1 with negative values."""
        input_tensor = torch.tensor([-0.5, -1.0, -1.5, -2.0], dtype=torch.float32)
        result = bessel_i1(input_tensor)
        expected = torch.from_numpy(scipy_i1(input_tensor.numpy()))

        torch.testing.assert_close(result, expected)

        # I₁(x) is an odd function, so I₁(-x) = -I₁(x)
        positive_input = torch.abs(input_tensor)
        positive_result = bessel_i1(positive_input)
        self.assertTrue(torch.allclose(-positive_result, result))

    def test_zero_values(self):
        """Test bessel_i1 with zero values."""
        input_tensor = torch.tensor([0.0], dtype=torch.float32)
        result = bessel_i1(input_tensor)

        # I₁(0) should be 0
        self.assertTrue(torch.allclose(result, torch.tensor([0.0])))

    def test_large_values(self):
        """Test bessel_i1 with large positive and negative values."""
        input_tensor = torch.tensor([10.0, -10.0, 20.0, -20.0], dtype=torch.float32)
        result = bessel_i1(input_tensor)
        expected = torch.from_numpy(scipy_i1(input_tensor.numpy()))

        # For large values, I₁(x) grows exponentially
        torch.testing.assert_close(result, expected)

        # Check that I₁(x) = -I₁(-x) for large values too
        self.assertTrue(torch.allclose(result[0], -result[1]))
        self.assertTrue(torch.allclose(result[2], -result[3]))

        # Check that I₁(x) grows with |x|
        self.assertTrue(torch.abs(result[2]) > torch.abs(result[0]))

    def test_small_values(self):
        """Test bessel_i1 with small values close to zero."""
        input_tensor = torch.tensor([1e-10, -1e-10, 1e-5, -1e-5], dtype=torch.float64)
        result = bessel_i1(input_tensor)
        expected = torch.from_numpy(scipy_i1(input_tensor.numpy()))

        # For very small values, I₁(x) ≈ x/2
        torch.testing.assert_close(result, expected)

        # For small x, I₁(x) ≈ x/2
        approx_result = input_tensor / 2
        self.assertTrue(torch.allclose(result, approx_result, atol=1e-10))

    def test_multi_dimensional_tensors(self):
        """Test bessel_i1 with multi-dimensional tensors."""
        input_tensor = torch.tensor([[0.0, 0.5], [1.0, 1.5]], dtype=torch.float32)
        result = bessel_i1(input_tensor)
        expected = torch.from_numpy(scipy_i1(input_tensor.numpy()))

        # Check shape
        self.assertEqual(result.shape, input_tensor.shape)

        # Check values
        torch.testing.assert_close(result, expected)

    def test_non_tensor_input(self):
        """Test bessel_i1 with non-tensor input (should raise TypeError)."""
        with self.assertRaises(TypeError):
            # List is not a tensor
            bessel_i1([0.0, 0.5, 1.0])

    def test_compare_with_scipy_implementation(self):
        """Test that bessel_i1 matches scipy's i1 implementation across a range of values."""
        # Create a range of values to test
        input_tensor = torch.linspace(-5.0, 5.0, 100, dtype=torch.float64)

        # Calculate with our function
        result = bessel_i1(input_tensor)

        # Calculate with scipy directly
        expected = torch.from_numpy(scipy_i1(input_tensor.numpy()))

        # Check that they match closely
        torch.testing.assert_close(result, expected)

        # Additional check for dtype preservation
        self.assertEqual(result.dtype, input_tensor.dtype)

        # Check the antisymmetry property: I₁(-x) = -I₁(x)
        negative_input = -input_tensor
        negative_result = bessel_i1(negative_input)
        self.assertTrue(torch.allclose(-result, negative_result))

    def test_relationship_with_i0(self):
        """Test the relationship between I₁ and I₀."""
        from scipy.special import i0 as bessel_i0

        # Test across a range of values, avoiding x close to 0 (division by zero)
        input_tensor = torch.linspace(0.1, 5.0, 100, dtype=torch.float64)

        # Calculate I₁(x) and I₀(x)
        i1_result = bessel_i1(input_tensor)
        i0_result = bessel_i0(input_tensor)

        # Check the relationship: I₁'(x) = I₀(x) - I₁(x)/x
        # We can approximate the derivative using central differences
        h = 1e-5
        x_plus_h = input_tensor + h
        x_minus_h = input_tensor - h
        i1_plus_h = bessel_i1(x_plus_h)
        i1_minus_h = bessel_i1(x_minus_h)
        i1_derivative = (i1_plus_h - i1_minus_h) / (2 * h)

        # Compute I₀(x) - I₁(x)/x
        expected_derivative = i0_result - i1_result / input_tensor

        # Check that they're close
        torch.testing.assert_close(
            i1_derivative, expected_derivative, rtol=1e-3, atol=1e-3
        )

    def test_bessel_i1_properties(self):
        """Test mathematical properties of the bessel_i1 function."""
        # I₁(x) is an odd function: I₁(-x) = -I₁(x)
        x_values = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
        neg_x_values = -x_values

        i1_x = bessel_i1(x_values)
        i1_neg_x = bessel_i1(neg_x_values)

        torch.testing.assert_close(i1_neg_x, -i1_x)

        # I₁(0) = 0
        self.assertTrue(
            torch.isclose(
                bessel_i1(torch.tensor([0.0], dtype=torch.float64))[0],
                torch.tensor(0.0, dtype=torch.float64),
            )
        )

        # I₁(x) is strictly increasing for x > 0
        x_increasing = torch.linspace(0.0, 5.0, 100, dtype=torch.float64)
        i1_x_increasing = bessel_i1(x_increasing)

        for i in range(1, len(x_increasing)):
            if x_increasing[i] > 0 and x_increasing[i - 1] > 0:
                self.assertTrue(i1_x_increasing[i] > i1_x_increasing[i - 1])

        # For x > 0, I₁(x) < I₀(x)
        from scipy.special import i0 as bessel_i0


input_tensor = torch.linspace(0, 10, steps=10)
expected_result = torch.Tensor(
    [
        0.0000e00,
        6.4581e-01,
        1.9536e00,
        5.3391e00,
        1.4628e01,
        4.0623e01,
        1.1420e02,
        3.2423e02,
        9.2770e02,
        2.6710e03,
    ]
)
assert torch.allclose(bessel_i1(input_tensor), expected_result, rtol=1e-3, atol=1e-3)
