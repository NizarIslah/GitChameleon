# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_8 import erfc
from scipy.special import erfc as scipy_erfc


class TestErfc(unittest.TestCase):
    """Test cases for the erfc function in sample_8.py."""

    def test_basic_values(self):
        """Test erfc with basic values."""
        input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
        result = erfc(input_tensor)
        expected = torch.from_numpy(scipy_erfc(input_tensor.numpy()))

        # Check that the result is a tensor
        self.assertIsInstance(result, torch.Tensor)

        # Check that the values match the expected output
        torch.testing.assert_close(result, expected)

        # Check that the dtype is preserved
        self.assertEqual(result.dtype, input_tensor.dtype)

        # erfc(0) should be 1
        self.assertTrue(
            torch.isclose(result[2], torch.tensor(1.0, dtype=torch.float32))
        )

        # erfc(-x) + erfc(x) = 2
        self.assertTrue(
            torch.isclose(result[0] + result[4], torch.tensor(2.0, dtype=torch.float32))
        )
        self.assertTrue(
            torch.isclose(result[1] + result[3], torch.tensor(2.0, dtype=torch.float32))
        )

    def test_large_values(self):
        """Test erfc with large positive and negative values."""
        input_tensor = torch.tensor([5.0, -5.0, 10.0, -10.0], dtype=torch.float32)
        result = erfc(input_tensor)
        expected = torch.from_numpy(scipy_erfc(input_tensor.numpy()))

        # For large values, erfc approaches 0 or 2
        torch.testing.assert_close(result, expected)

        # erfc(x) approaches 0 as x approaches infinity
        self.assertTrue(
            torch.isclose(result[0], torch.tensor(0.0, dtype=torch.float32), atol=1e-4)
        )
        self.assertTrue(
            torch.isclose(result[2], torch.tensor(0.0, dtype=torch.float32), atol=1e-9)
        )

        # erfc(x) approaches 2 as x approaches negative infinity
        self.assertTrue(
            torch.isclose(result[1], torch.tensor(2.0, dtype=torch.float32), atol=1e-4)
        )
        self.assertTrue(
            torch.isclose(result[3], torch.tensor(2.0, dtype=torch.float32), atol=1e-9)
        )

    def test_small_values(self):
        """Test erfc with small values close to zero."""
        input_tensor = torch.tensor(
            [-1e-5, -1e-10, 0.0, 1e-10, 1e-5], dtype=torch.float64
        )
        result = erfc(input_tensor)
        expected = torch.from_numpy(scipy_erfc(input_tensor.numpy()))

        # For small values, erfc(x) ≈ 1 - (2/√π) * x
        torch.testing.assert_close(result, expected)

        # For very small x, erfc(x) ≈ 1 - (2/√π) * x
        approx_factor = 2.0 / np.sqrt(np.pi)
        small_x = input_tensor[3]  # 1e-10
        expected_approx = 1.0 - small_x * approx_factor
        self.assertTrue(
            torch.isclose(result[3], expected_approx, rtol=1e-10, atol=1e-10)
        )

        # erfc(0) = 1
        self.assertTrue(
            torch.isclose(result[2], torch.tensor(1.0, dtype=torch.float64))
        )

    def test_multi_dimensional_tensors(self):
        """Test erfc with multi-dimensional tensors."""
        input_tensor = torch.tensor([[-2.0, -1.0], [0.0, 1.0]], dtype=torch.float32)
        result = erfc(input_tensor)
        expected = torch.from_numpy(scipy_erfc(input_tensor.numpy()))

        # Check shape
        self.assertEqual(result.shape, input_tensor.shape)

        # Check values
        torch.testing.assert_close(result, expected)

    def test_non_tensor_input(self):
        """Test erfc with non-tensor input (should raise TypeError)."""
        with self.assertRaises(TypeError):
            # List is not a tensor
            erfc([-2.0, -1.0, 0.0, 1.0, 2.0])

    def test_different_dtypes(self):
        """Test erfc with different dtypes."""
        # Test with float32
        input_tensor_f32 = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        result_f32 = erfc(input_tensor_f32)
        self.assertEqual(result_f32.dtype, torch.float32)

        # Test with float64
        input_tensor_f64 = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        result_f64 = erfc(input_tensor_f64)
        self.assertEqual(result_f64.dtype, torch.float64)

    def test_compare_with_scipy_implementation(self):
        """Test that erfc matches scipy's erfc implementation across a range of values."""
        # Create a range of values to test
        input_tensor = torch.linspace(-3.0, 3.0, 100, dtype=torch.float64)

        # Calculate with our function
        result = erfc(input_tensor)

        # Calculate with scipy directly
        expected = torch.from_numpy(scipy_erfc(input_tensor.numpy()))

        # Check that they match closely
        torch.testing.assert_close(result, expected)

        # Additional check for dtype preservation
        self.assertEqual(result.dtype, input_tensor.dtype)

        # Check that erfc(-x) + erfc(x) = 2 for all x
        negative_input = -input_tensor
        negative_result = erfc(negative_input)
        sum_result = result + negative_result
        self.assertTrue(torch.allclose(sum_result, torch.full_like(sum_result, 2.0)))

        # Check that erfc is bounded between 0 and 2
        self.assertTrue(torch.all(result >= 0.0))
        self.assertTrue(torch.all(result <= 2.0))

    def test_relationship_with_erf(self):
        """Test the relationship between erfc and erf: erfc(x) = 1 - erf(x)."""
        from scipy.special import erf


input_tensor = torch.linspace(0, 10, steps=10)
expected_result = torch.Tensor(
    [
        1.0000e00,
        1.1610e-01,
        1.6740e-03,
        2.4285e-06,
        3.2702e-10,
        3.9425e-15,
        4.1762e-21,
        3.8452e-28,
        3.0566e-36,
        1.4013e-45,
    ]
)
assert torch.allclose(erfc(input_tensor), expected_result, rtol=1e-3, atol=1e-3)
