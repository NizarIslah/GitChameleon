import json

# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np

# from flask import Flask # Flask class not directly instantiated here, app object is imported

# The original sys.path.append logic is assumed to be correct for the project structure
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sample_163 import MyCustomJSONHandler


class TestSample163(unittest.TestCase):
    def test_custom_json_encoder(self):
        """Test the custom JSON encoder directly"""
        # This test calls the encoder directly, bypassing flask.jsonify,
        # so it should work as expected once assertions are correct.
        encoder = MyCustomJSONHandler()

        # Test with regular array
        test_arr = np.array([1, 2, 3, 4, 5])
        encoded = encoder.default(test_arr)
        # MyCustomJSONHandler promotes integers to floats.
        self.assertEqual(encoded, [1.0, 2.0, 3.0, 4.0, 5.0])

        # Test with array containing duplicates
        test_arr = np.array([1, 2, 2, 3, 3, 3])
        encoded = encoder.default(test_arr)
        # MyCustomJSONHandler uniques values and promotes to float.
        self.assertEqual(encoded, [1.0, 2.0, 3.0])

        # Test with array containing NaN values
        test_arr = np.array([1.0, 2.0, np.nan, 3.0, np.nan])
        encoded = encoder.default(
            test_arr
        )  # Handler output: [1.0, 2.0, 3.0, np.nan, np.nan]

        # Assertions based on the handler's behavior of preserving all NaNs.
        self.assertEqual(len(encoded), 5)
        self.assertEqual(encoded[:3], [1.0, 2.0, 3.0])  # Unique non-NaN values
        self.assertTrue(np.isnan(encoded[3]))  # First original NaN
        self.assertTrue(np.isnan(encoded[4]))  # Second original NaN

        # Test with non-numpy object (should raise TypeError)
        with self.assertRaises(TypeError):
            encoder.default("not a numpy array")


if __name__ == "__main__":
    unittest.main()
