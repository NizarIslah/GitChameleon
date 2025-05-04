# Add the parent directory to import sys
import os
import sys
import unittest

import flask
import numpy as np
from werkzeug.exceptions import NotFound

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_167 import stack_and_save


class TestStackAndSave(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.base_path = "/tmp/test_base"
        os.makedirs(self.base_path, exist_ok=True)

        # Create some test arrays
        self.int_array = np.array([[1, 2], [3, 4]], dtype=np.int32)
        self.float_array = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        self.double_array = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)

    def tearDown(self):
        # Clean up the temporary directory
        if os.path.exists(self.base_path):
            import shutil

            shutil.rmtree(self.base_path)

    def test_safe_path_joining(self):
        """Test that paths are joined safely."""
        # Valid sub path
        sub_path = "valid_subdir"
        joined_path, _ = stack_and_save(
            [self.float_array], self.base_path, sub_path, "safe", np.float32
        )
        self.assertEqual(joined_path, os.path.join(self.base_path, sub_path))

        # Attempt to use a path that tries to escape the base directory
        with self.assertRaises(NotFound):
            stack_and_save(
                [self.float_array],
                self.base_path,
                "../escape_attempt",
                "safe",
                np.float32,
            )

    def test_safe_casting(self):
        """Test safe casting policy."""
        # Safe casting from float32 to float64 should work
        _, stacked = stack_and_save(
            [self.float_array], self.base_path, "test", "safe", np.float64
        )
        self.assertEqual(stacked.dtype, np.float64)

        # Safe casting from float64 to float32 should fail
        with self.assertRaises(TypeError):
            stack_and_save(
                [self.double_array], self.base_path, "test", "safe", np.float32
            )

    def test_unsafe_casting(self):
        """Test unsafe casting policy."""
        # Unsafe casting from float64 to float32 should work
        _, stacked = stack_and_save(
            [self.double_array], self.base_path, "test", "unsafe", np.float32
        )
        self.assertEqual(stacked.dtype, np.float32)

        # Unsafe casting from int32 to float32 should work
        _, stacked = stack_and_save(
            [self.int_array], self.base_path, "test", "unsafe", np.float32
        )
        self.assertEqual(stacked.dtype, np.float32)

    def test_stacking_multiple_arrays(self):
        """Test stacking multiple arrays."""
        # Stack multiple float32 arrays to float32
        arrays = [
            np.array([[1.0, 2.0]], dtype=np.float32),
            np.array([[3.0, 4.0]], dtype=np.float32),
            np.array([[5.0, 6.0]], dtype=np.float32),
        ]

        _, stacked = stack_and_save(arrays, self.base_path, "test", "safe", np.float32)

        # Check dimensions and content
        self.assertEqual(stacked.shape, (3, 2))
        np.testing.assert_array_equal(
            stacked, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        )

    def test_mixed_array_types(self):
        """Test with mixed array types that can be safely cast."""
        arrays = [
            np.array([[1, 2]], dtype=np.int32),
            np.array([[3.0, 4.0]], dtype=np.float32),
        ]

        # Both int32 and float32 can be safely cast to float64
        _, stacked = stack_and_save(arrays, self.base_path, "test", "safe", np.float64)

        self.assertEqual(stacked.dtype, np.float64)
        self.assertEqual(stacked.shape, (2, 2))
        np.testing.assert_array_equal(
            stacked, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        )


if __name__ == "__main__":
    unittest.main()
