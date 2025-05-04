# Add the parent directory to import sys
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import werkzeug.exceptions

sys.path.append(str(Path(__file__).parent.parent))
from sample_168 import error404, stack_and_save


class TestStackAndSave(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = self.temp_dir.name

        # Create some test arrays
        self.arr1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        self.arr2 = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)
        self.arr_list = [self.arr1, self.arr2]

    def tearDown(self):
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    def test_successful_stack_and_save(self):
        """Test successful stacking and path joining."""
        sub_path = "test_dir/test_file.npy"
        # Create the subdirectory
        os.makedirs(os.path.join(self.base_path, "test_dir"), exist_ok=True)

        # Test with safe casting and float32 dtype
        joined_path, stacked = stack_and_save(
            self.arr_list, self.base_path, sub_path, "safe", np.float32
        )

        # Check the joined path
        expected_path = os.path.join(self.base_path, sub_path)
        self.assertEqual(joined_path, expected_path)

        # Check the stacked array
        expected_stack = np.vstack(self.arr_list)
        np.testing.assert_array_equal(stacked, expected_stack)
        self.assertEqual(stacked.dtype, np.float32)

    def test_unsafe_casting(self):
        """Test with unsafe casting policy."""
        sub_path = "test_file.npy"

        # Test with unsafe casting and float64 dtype
        joined_path, stacked = stack_and_save(
            self.arr_list, self.base_path, sub_path, "unsafe", np.float64
        )

        # Check the stacked array dtype
        self.assertEqual(stacked.dtype, np.float64)

    def test_path_traversal_error(self):
        """Test that path traversal attempts raise a 404 error."""
        # Try to access a path outside the base directory
        with self.assertRaises(error404):
            stack_and_save(
                self.arr_list, self.base_path, "../../../etc/passwd", "safe", np.float32
            )

    def test_incompatible_dtype_error(self):
        """Test that incompatible dtype with casting policy raises TypeError."""
        sub_path = "test_file.npy"

        # Create arrays with int dtype
        int_arr1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        int_arr2 = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int32)
        int_arr_list = [int_arr1, int_arr2]

        # Using safe casting with float32 should raise TypeError when casting from int to float
        with self.assertRaises(TypeError):
            stack_and_save(
                int_arr_list,
                self.base_path,
                sub_path,
                "safe",
                np.float16,  # Using float16 to potentially trigger casting error
            )


if __name__ == "__main__":
    unittest.main()
