import math

# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_12 import log_ndtr


class TestLogNdtr(unittest.TestCase):
    """Test cases for the log_ndtr function in sample_12.py."""

    def test_basic_functionality_positive_values(self):
        """Test log_ndtr with basic positive values."""
        # Create a tensor with positive values
        input_tensor = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
        result = log_ndtr(input_tensor)

        # Check that the result is a tensor
        self.assertIsInstance(result, torch.Tensor)

        # Check that the shape is preserved
        self.assertEqual(result.shape, input_tensor.shape)

        # For positive values, log_ndtr should be close to log(0.5 + 0.5*erf(x/sqrt(2)))
        # We'll compare with manually calculated values
        expected = torch.log(0.5 + 0.5 * torch.erf(input_tensor / math.sqrt(2)))
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_negative_values(self):
        """Test log_ndtr with negative values."""
        input_tensor = torch.tensor([-0.5, -1.0, -2.0], dtype=torch.float32)
        result = log_ndtr(input_tensor)

        # Check that the shape is preserved
        self.assertEqual(result.shape, input_tensor.shape)

        # For negative values, log_ndtr should still return valid results
        # We'll compare with manually calculated values
        expected = torch.log(0.5 - 0.5 * torch.erf(-input_tensor / math.sqrt(2)))
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_zero_value(self):
        """Test log_ndtr with zero value."""
        input_tensor = torch.tensor([0.0], dtype=torch.float32)
        result = log_ndtr(input_tensor)

        # log_ndtr(0) should be log(0.5)
        expected = torch.log(torch.tensor([0.5]))
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_multi_dimensional_tensors(self):
        """Test log_ndtr with multi-dimensional tensors."""
        input_tensor = torch.tensor([[0.5, 1.0], [1.5, 2.0]], dtype=torch.float32)
        result = log_ndtr(input_tensor)

        # Check that the shape is preserved
        self.assertEqual(result.shape, input_tensor.shape)

        # Check values
        expected = torch.log(0.5 + 0.5 * torch.erf(input_tensor / math.sqrt(2)))
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_different_dtypes(self):
        """Test log_ndtr with different dtypes."""
        # Test with float32
        input_float32 = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
        result_float32 = log_ndtr(input_float32)
        self.assertEqual(result_float32.dtype, torch.float32)

        # Test with float64
        input_float64 = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        result_float64 = log_ndtr(input_float64)
        self.assertEqual(result_float64.dtype, torch.float64)

        # Values should be similar regardless of dtype
        torch.testing.assert_close(
            result_float32, result_float64.to(torch.float32), rtol=1e-5, atol=1e-5
        )

    def test_extreme_values(self):
        """Test log_ndtr with extreme values."""
        # Very large positive values
        large_positive = torch.tensor([10.0, 20.0, 50.0], dtype=torch.float32)
        result_large = log_ndtr(large_positive)

        # For large positive values, log_ndtr(x) approaches 0
        # (since ndtr(x) approaches 1)
        # The values should be close to 0 but still negative
        self.assertTrue(torch.all(result_large <= 0))

        # Very large negative values
        large_negative = torch.tensor([-10.0, -20.0, -50.0], dtype=torch.float32)
        result_negative = log_ndtr(large_negative)

        # For large negative values, log_ndtr(x) should be very negative
        # (since ndtr(x) approaches 0)
        self.assertTrue(torch.all(result_negative < -10))

        # The values should decrease as the input becomes more negative
        self.assertTrue(result_negative[0] > result_negative[1])
        self.assertTrue(result_negative[1] > result_negative[2])

    def test_non_tensor_input(self):
        """Test log_ndtr with non-tensor input (should raise TypeError)."""
        with self.assertRaises(TypeError):
            # List is not a tensor
            log_ndtr([0.5, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
