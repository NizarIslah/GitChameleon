# library: lightgbm
# version: 3.0.0
# extra_dependencies: ['numpy==1.26.4']
import unittest
import numpy as np
import json
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_87 import dump_json


class TestDumpJson(unittest.TestCase):
    """Test cases for the dump_json function."""

    def test_basic_types(self):
        """Test dump_json with basic Python types."""
        # Test with string
        self.assertEqual(dump_json("test"), '"test"')

        # Test with integer
        self.assertEqual(dump_json(42), "42")

        # Test with float
        self.assertEqual(dump_json(3.14), "3.14")

        # Test with boolean
        self.assertEqual(dump_json(True), "true")
        self.assertEqual(dump_json(False), "false")

        # Test with None
        self.assertEqual(dump_json(None), "null")

    def test_container_types(self):
        """Test dump_json with container types."""
        # Test with list
        self.assertEqual(dump_json([1, 2, 3]), "[1, 2, 3]")

        # Test with dictionary
        result = dump_json({"a": 1, "b": 2})
        # Since dictionary order is not guaranteed in all Python versions
        self.assertTrue(result == '{"a": 1, "b": 2}' or result == '{"b": 2, "a": 1}')

        # Test with nested structures
        expected = '{"data": [1, 2, 3], "info": {"name": "test"}}'
        result = dump_json({"data": [1, 2, 3], "info": {"name": "test"}})
        # Parse both to compare as objects since string order might differ
        self.assertEqual(json.loads(result), json.loads(expected))

    def test_numpy_types(self):
        """Test dump_json with NumPy types."""
        # Test with numpy scalar
        np_int = np.int64(42)
        self.assertEqual(dump_json(np_int), "42")

        np_float = np.float64(3.14)
        self.assertEqual(dump_json(np_float), "3.14")

        # Test with numpy array
        np_array = np.array([1, 2, 3])
        self.assertEqual(dump_json(np_array), "[1, 2, 3]")

        # Test with 2D numpy array
        np_2d_array = np.array([[1, 2], [3, 4]])
        self.assertEqual(json.loads(dump_json(np_2d_array)), [[1, 2], [3, 4]])

        # Test with mixed numpy and Python types
        mixed_data = {
            "array": np.array([1, 2, 3]),
            "value": np.float32(2.5),
            "regular": "string",
        }
        result = dump_json(mixed_data)
        parsed = json.loads(result)
        self.assertEqual(parsed["array"], [1, 2, 3])
        self.assertAlmostEqual(parsed["value"], 2.5, places=5)
        self.assertEqual(parsed["regular"], "string")


if __name__ == "__main__":
    unittest.main()
