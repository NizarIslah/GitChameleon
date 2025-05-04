# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_7 import erf
from scipy.special import erf as scipy_erf


class TestErf(unittest.TestCase):
    """Test cases for the erf function in sample_7.py."""

    def test_basic_values(self):
        """Test erf with basic values."""
        input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
        result = erf(input_tensor)
        expected = torch.from_numpy(scipy_erf(input_tensor.numpy()))

        # Check that the result is a tensor
        self.assertIsInstance(result, torch.Tensor)

        # Check that the values match the expected output
        torch.testing.assert_close(result, expected)

        # Check that the dtype is preserved
        self.assertEqual(result.dtype, input_tensor.dtype)

        # erf(0) should be 0
        self.assertTrue(
            torch.isclose(result[2], torch.tensor(0.0, dtype=torch.float32))
        )

        # erf is an odd function, so erf(-x) = -erf(x)
        self.assertTrue(torch.isclose(result[0], -result[4]))
        self.assertTrue(torch.isclose(result[1], -result[3]))

    def test_large_values(self):
        """Test erf with large positive and negative values."""
        input_tensor = torch.tensor([5.0, -5.0, 10.0, -10.0], dtype=torch.float32)
        result = erf(input_tensor)
        expected = torch.from_numpy(scipy_erf(input_tensor.numpy()))

        # For large values, erf approaches 1 or -1
        torch.testing.assert_close(result, expected)

        # erf(x) approaches 1 as x approaches infinity
        self.assertTrue(
            torch.isclose(result[0], torch.tensor(1.0, dtype=torch.float32), atol=1e-4)
        )
        self.assertTrue(
            torch.isclose(result[2], torch.tensor(1.0, dtype=torch.float32), atol=1e-9)
        )

        # erf(x) approaches -1 as x approaches negative infinity
        self.assertTrue(
            torch.isclose(result[1], torch.tensor(-1.0, dtype=torch.float32), atol=1e-4)
        )
        self.assertTrue(
            torch.isclose(result[3], torch.tensor(-1.0, dtype=torch.float32), atol=1e-9)
        )

    def test_small_values(self):
        """Test erf with small values close to zero."""
        input_tensor = torch.tensor(
            [-1e-5, -1e-10, 0.0, 1e-10, 1e-5], dtype=torch.float64
        )
        result = erf(input_tensor)
        expected = torch.from_numpy(scipy_erf(input_tensor.numpy()))

        # For small values, erf(x) ≈ (2/√π) * x
        torch.testing.assert_close(result, expected)

        # For very small x, erf(x) ≈ (2/√π) * x
        approx_factor = 2.0 / np.sqrt(np.pi)
        small_x = input_tensor[3]  # 1e-10
        expected_approx = small_x * approx_factor
        self.assertTrue(
            torch.isclose(result[3], expected_approx, rtol=1e-10, atol=1e-10)
        )

    def test_multi_dimensional_tensors(self):
        """Test erf with multi-dimensional tensors."""
        input_tensor = torch.tensor([[-2.0, -1.0], [0.0, 1.0]], dtype=torch.float32)
        result = erf(input_tensor)
        expected = torch.from_numpy(scipy_erf(input_tensor.numpy()))

        # Check shape
        self.assertEqual(result.shape, input_tensor.shape)

        # Check values
        torch.testing.assert_close(result, expected)

    def test_non_tensor_input(self):
        """Test erf with non-tensor input (should raise TypeError)."""
        with self.assertRaises(TypeError):
            # List is not a tensor
            erf([-2.0, -1.0, 0.0, 1.0, 2.0])

    def test_different_dtypes(self):
        """Test erf with different dtypes."""
        # Test with float32
        input_tensor_f32 = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        result_f32 = erf(input_tensor_f32)
        self.assertEqual(result_f32.dtype, torch.float32)

        # Test with float64
        input_tensor_f64 = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        result_f64 = erf(input_tensor_f64)
        self.assertEqual(result_f64.dtype, torch.float64)

    def test_compare_with_scipy_implementation(self):
        """Test that erf matches scipy's erf implementation across a range of values."""
        # Create a range of values to test
        input_tensor = torch.linspace(-3.0, 3.0, 100, dtype=torch.float64)

        # Calculate with our function
        result = erf(input_tensor)

        # Calculate with scipy directly
        expected = torch.from_numpy(scipy_erf(input_tensor.numpy()))

        # Check that they match closely
        torch.testing.assert_close(result, expected)

        # Additional check for dtype preservation
        self.assertEqual(result.dtype, input_tensor.dtype)

        # Check the odd function property: erf(-x) = -erf(x)
        negative_input = -input_tensor
        negative_result = erf(negative_input)
        self.assertTrue(torch.allclose(-result, negative_result))

        # Check that erf is bounded between -1 and 1
        self.assertTrue(torch.all(result >= -1.0))
        self.assertTrue(torch.all(result <= 1.0))

    def test_erf_properties(self):
        """Test mathematical properties of the erf function."""
        # erf is an odd function: erf(-x) = -erf(x)
        x_values = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
        neg_x_values = -x_values

        erf_x = erf(x_values)
        erf_neg_x = erf(neg_x_values)

        torch.testing.assert_close(erf_neg_x, -erf_x)

        # erf(0) = 0
        self.assertTrue(
            torch.isclose(
                erf(torch.tensor([0.0], dtype=torch.float64))[0],
                torch.tensor(0.0, dtype=torch.float64),
            )
        )

        # erf is strictly increasing
        x_increasing = torch.linspace(-3.0, 3.0, 100, dtype=torch.float64)
        erf_x_increasing = erf(x_increasing)

        for i in range(1, len(x_increasing)):
            self.assertTrue(erf_x_increasing[i] > erf_x_increasing[i - 1])


if __name__ == "__main__":
    unittest.main()
