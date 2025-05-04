import json
import os
import sys
import unittest

import flask
import numpy as np
from scipy import linalg

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_170 import MyCustomJSONHandler, app, data, eval


class TestSample170(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.testing = True
        self.client = self.app.test_client()

    def test_data_route(self):
        """Test the /data route with a simple array"""
        # Create a test array
        test_arr = [1, 2, 3, 4, 5]

        # Test using the eval function
        result = eval(self.app, data, test_arr)
        result_json = json.loads(result)

        self.assertIn("numbers", result_json)
        self.assertEqual(result_json["numbers"], test_arr)

    def test_custom_json_encoder_with_other_objects(self):
        """Test the custom JSON encoder with objects it doesn't handle specially"""
        encoder = MyCustomJSONHandler()

        # Test with a regular list
        with self.assertRaises(TypeError):
            encoder.default([1, 2, 3])

        # Test with a 2D ndarray
        with self.assertRaises(TypeError):
            encoder.default(np.array([[1, 2], [3, 4]]))

        # Test with a 3D ndarray where last two dimensions are not equal
        with self.assertRaises(TypeError):
            encoder.default(np.zeros((2, 3, 4)))


if __name__ == "__main__":
    unittest.main()
