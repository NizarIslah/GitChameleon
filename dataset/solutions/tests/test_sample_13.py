# Add the parent directory to import sys
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_13 import invert_mask


class TestInvertMask(unittest.TestCase):
    """Test cases for the invert_mask function in sample_13.py."""

    def test_basic_comparison(self):
        """Test invert_mask with basic tensor comparison."""
        tensor1 = torch.tensor([1, 2, 3, 4, 5])
        tensor2 = torch.tensor([3, 3, 3, 3, 3])
        result = invert_mask(tensor1, tensor2)

        # Expected: ~(tensor1 < tensor2).bool() = ~[True, True, False, False, False] = [False, False, True, True, True]
        expected = torch.tensor([False, False, True, True, True])

        # Check that the result is a boolean tensor
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.bool)

        # Check that the values match the expected output
        torch.testing.assert_close(result, expected)

        # Check that the shape is preserved
        self.assertEqual(result.shape, tensor1.shape)

    def test_equal_values(self):
        """Test invert_mask when values are equal."""
        tensor1 = torch.tensor([3, 3, 3])
        tensor2 = torch.tensor([3, 3, 3])
        result = invert_mask(tensor1, tensor2)

        # When values are equal, tensor1 < tensor2 is False, so ~False is True
        expected = torch.tensor([True, True, True])

        torch.testing.assert_close(result, expected)

    def test_all_less_than(self):
        """Test invert_mask when all values in tensor1 are less than tensor2."""
        tensor1 = torch.tensor([1, 2, 3])
        tensor2 = torch.tensor([4, 5, 6])
        result = invert_mask(tensor1, tensor2)

        # All values in tensor1 are less than tensor2, so all are False after inversion
        expected = torch.tensor([False, False, False])

        torch.testing.assert_close(result, expected)

    def test_all_greater_than(self):
        """Test invert_mask when all values in tensor1 are greater than tensor2."""
        tensor1 = torch.tensor([4, 5, 6])
        tensor2 = torch.tensor([1, 2, 3])
        result = invert_mask(tensor1, tensor2)

        # All values in tensor1 are greater than tensor2, so all are True after inversion
        expected = torch.tensor([True, True, True])

        torch.testing.assert_close(result, expected)

    def test_multi_dimensional_tensors(self):
        """Test invert_mask with multi-dimensional tensors."""
        tensor1 = torch.tensor([[1, 5], [3, 4]])
        tensor2 = torch.tensor([[2, 2], [2, 2]])
        result = invert_mask(tensor1, tensor2)

        # Expected: ~([[True, False], [False, False]]) = [[False, True], [True, True]]
        expected = torch.tensor([[False, True], [True, True]])

        # Check shape
        self.assertEqual(result.shape, tensor1.shape)

        # Check values
        torch.testing.assert_close(result, expected)

    def test_different_dtypes(self):
        """Test invert_mask with different dtypes."""
        # Test with float32
        tensor1_f32 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        tensor2_f32 = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)
        result_f32 = invert_mask(tensor1_f32, tensor2_f32)

        # Result should always be boolean regardless of input dtype
        self.assertEqual(result_f32.dtype, torch.bool)

        # Test with int64
        tensor1_i64 = torch.tensor([1, 2, 3], dtype=torch.int64)
        tensor2_i64 = torch.tensor([2, 2, 2], dtype=torch.int64)
        result_i64 = invert_mask(tensor1_i64, tensor2_i64)

        self.assertEqual(result_i64.dtype, torch.bool)

        # Results should be the same regardless of dtype
        torch.testing.assert_close(result_f32, result_i64)

    def test_mixed_dtypes(self):
        """Test invert_mask with mixed dtypes."""
        tensor1 = torch.tensor([1, 2, 3], dtype=torch.int32)
        tensor2 = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)

        # PyTorch handles type promotion automatically
        result = invert_mask(tensor1, tensor2)
        expected = torch.tensor([False, True, True])

        torch.testing.assert_close(result, expected)

    def test_non_tensor_input(self):
        """Test invert_mask with non-tensor input (should raise TypeError)."""
        with self.assertRaises(TypeError):
            # List is not a tensor
            invert_mask([1, 2, 3], torch.tensor([2, 2, 2]))

        with self.assertRaises(TypeError):
            # List is not a tensor
            invert_mask(torch.tensor([1, 2, 3]), [2, 2, 2])

    def test_boolean_input(self):
        """Test invert_mask with boolean input tensors."""
        tensor1 = torch.tensor([True, False, True])
        tensor2 = torch.tensor([False, True, True])
        result = invert_mask(tensor1, tensor2)

        # In boolean comparison, True > False
        # So tensor1 < tensor2 gives [False, True, False]
        # After inversion: [True, False, True]
        expected = torch.tensor([True, False, True])

        torch.testing.assert_close(result, expected)

    def test_empty_tensors(self):
        """Test invert_mask with empty tensors."""
        tensor1 = torch.tensor([])
        tensor2 = torch.tensor([])
        result = invert_mask(tensor1, tensor2)

        # Empty tensors should return empty boolean tensor
        expected = torch.tensor([], dtype=torch.bool)

        self.assertEqual(result.numel(), 0)
        self.assertEqual(result.dtype, torch.bool)


if __name__ == "__main__":
    unittest.main()
