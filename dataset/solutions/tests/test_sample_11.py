# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_11 import invert_mask


class TestInvertMask(unittest.TestCase):
    """Test cases for the invert_mask function in sample_11.py."""

    def test_basic_comparison(self):
        """Test invert_mask with basic tensor comparisons."""
        tensor1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
        tensor2 = torch.tensor([3, 3, 3, 3, 3], dtype=torch.float32)
        result = invert_mask(tensor1, tensor2)

        # Check that the result is a tensor
        self.assertIsInstance(result, torch.Tensor)

        # Check that the result is a boolean tensor
        self.assertEqual(result.dtype, torch.bool)

        # Check the values: ~(tensor1 < tensor2) should be True where tensor1 >= tensor2
        expected = torch.tensor([False, False, True, True, True])
        self.assertTrue(torch.all(result == expected))

        # Check that the shape is preserved
        self.assertEqual(result.shape, tensor1.shape)

    def test_equal_values(self):
        """Test invert_mask with equal values."""
        tensor1 = torch.tensor([3, 3, 3], dtype=torch.float32)
        tensor2 = torch.tensor([3, 3, 3], dtype=torch.float32)
        result = invert_mask(tensor1, tensor2)

        # For equal values, tensor1 < tensor2 is False, so ~(False) is True
        expected = torch.tensor([True, True, True])
        self.assertTrue(torch.all(result == expected))

    def test_all_less_than(self):
        """Test invert_mask when all values in tensor1 are less than tensor2."""
        tensor1 = torch.tensor([1, 2, 3], dtype=torch.float32)
        tensor2 = torch.tensor([4, 5, 6], dtype=torch.float32)
        result = invert_mask(tensor1, tensor2)

        # All values in tensor1 are less than tensor2, so ~(True) is False
        expected = torch.tensor([False, False, False])
        self.assertTrue(torch.all(result == expected))

    def test_all_greater_than(self):
        """Test invert_mask when all values in tensor1 are greater than tensor2."""
        tensor1 = torch.tensor([4, 5, 6], dtype=torch.float32)
        tensor2 = torch.tensor([1, 2, 3], dtype=torch.float32)
        result = invert_mask(tensor1, tensor2)

        # All values in tensor1 are greater than tensor2, so ~(False) is True
        expected = torch.tensor([True, True, True])
        self.assertTrue(torch.all(result == expected))

    def test_mixed_comparison(self):
        """Test invert_mask with mixed comparison results."""
        tensor1 = torch.tensor([1, 3, 5, 7, 9], dtype=torch.float32)
        tensor2 = torch.tensor([2, 3, 4, 8, 8], dtype=torch.float32)
        result = invert_mask(tensor1, tensor2)

        # Expected: ~(tensor1 < tensor2) = ~[True, False, False, True, False] = [False, True, True, False, True]
        expected = torch.tensor([False, True, True, False, True])
        self.assertTrue(torch.all(result == expected))

    def test_different_dtypes(self):
        """Test invert_mask with different dtypes."""
        # Test with integer tensors
        tensor1_int = torch.tensor([1, 2, 3], dtype=torch.int32)
        tensor2_int = torch.tensor([2, 2, 2], dtype=torch.int32)
        result_int = invert_mask(tensor1_int, tensor2_int)

        expected_int = torch.tensor([False, True, True])
        self.assertTrue(torch.all(result_int == expected_int))

        # Test with float tensors
        tensor1_float = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
        tensor2_float = torch.tensor([2.5, 2.5, 2.5], dtype=torch.float32)
        result_float = invert_mask(tensor1_float, tensor2_float)

        expected_float = torch.tensor([False, True, True])
        self.assertTrue(torch.all(result_float == expected_float))

    def test_multi_dimensional_tensors(self):
        """Test invert_mask with multi-dimensional tensors."""
        tensor1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        tensor2 = torch.tensor([[2, 2], [2, 2]], dtype=torch.float32)
        result = invert_mask(tensor1, tensor2)

        # Check shape
        self.assertEqual(result.shape, tensor1.shape)

        # Check values
        expected = torch.tensor([[False, True], [True, True]])
        self.assertTrue(torch.all(result == expected))

    def test_broadcasting(self):
        """Test invert_mask with broadcasting."""
        tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        tensor2 = torch.tensor(
            [2, 3, 4], dtype=torch.float32
        )  # Will be broadcast to [[2, 3, 4], [2, 3, 4]]
        result = invert_mask(tensor1, tensor2)

        # Check shape
        self.assertEqual(result.shape, tensor1.shape)

        # Check values
        expected = torch.tensor([[False, False, False], [True, True, True]])
        self.assertTrue(torch.all(result == expected))

    def test_scalar_tensor(self):
        """Test invert_mask with a scalar tensor."""
        tensor1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
        tensor2 = torch.tensor(3, dtype=torch.float32)  # Scalar tensor
        result = invert_mask(tensor1, tensor2)

        # Check values
        expected = torch.tensor([False, False, True, True, True])
        self.assertTrue(torch.all(result == expected))

    def test_non_tensor_input(self):
        """Test invert_mask with non-tensor input (should raise TypeError)."""
        with self.assertRaises(TypeError):
            # List is not a tensor
            invert_mask([1, 2, 3], torch.tensor([2, 2, 2]))

        with self.assertRaises(TypeError):
            # List is not a tensor
            invert_mask(torch.tensor([1, 2, 3]), [2, 2, 2])


if __name__ == "__main__":
    unittest.main()
