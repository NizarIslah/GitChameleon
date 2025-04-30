import json
import os
import sys
import unittest

import numpy as np
from flask import Flask

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_163 import MyCustomJSONHandler, app, data, eval


class TestSample163(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def test_data_route(self):
        """Test the /data route with various numpy arrays and no NaNs."""
        # Test with regular array
        test_arr = np.array([1, 2, 3, 4, 5])
        with self.app.test_request_context():
            response = data(test_arr)
            response_data = json.loads(response.get_data(as_text=True))
            self.assertEqual(response_data['numbers'], [1, 2, 3, 4, 5])

        # Test with array containing duplicates
        test_arr = np.array([1, 2, 2, 3, 3, 3])
        with self.app.test_request_context():
            response = data(test_arr)
            response_data = json.loads(response.get_data(as_text=True))
            self.assertEqual(response_data['numbers'], [1, 2, 3])

    def test_data_with_nan(self):
        """
        Test the /data route with arrays containing NaN values.
        We expect duplicate values to be removed and all NaNs to be discarded.
        """
        test_arr = np.array([1.0, 2.0, np.nan, 3.0, np.nan])
        with self.app.test_request_context():
            response = data(test_arr)
            response_data = json.loads(response.get_data(as_text=True))
            # Expected unique non-NaN values only: [1.0, 2.0, 3.0]
            self.assertEqual(response_data['numbers'], [1.0, 2.0, 3.0])

    def test_eval_function(self):
        """Test the eval function."""
        test_arr = np.array([1, 2, 3, 4, 5])
        result = eval(self.app, data, test_arr)
        # Convert bytes to string and then to JSON
        result_json = json.loads(result.decode('utf-8'))
        self.assertEqual(result_json['numbers'], [1, 2, 3, 4, 5])

    def test_custom_json_encoder(self):
        """
        Test the custom JSON encoder directly.
        We ensure that duplicates are removed and NaN values are discarded entirely.
        """
        encoder = MyCustomJSONHandler()

        # Test with regular array
        test_arr = np.array([1, 2, 3, 4, 5])
        encoded = encoder.default(test_arr)
        self.assertEqual(encoded, [1, 2, 3, 4, 5])

        # Test with array containing duplicates
        test_arr = np.array([1, 2, 2, 3, 3, 3])
        encoded = encoder.default(test_arr)
        self.assertEqual(encoded, [1, 2, 3])

        # Test with array containing NaN values
        test_arr = np.array([1.0, 2.0, np.nan, 3.0, np.nan])
        encoded = encoder.default(test_arr)
        # We expect [1.0, 2.0, 3.0]
        self.assertEqual(encoded, [1.0, 2.0, 3.0])

        # Test with non-numpy object (should raise TypeError)
        with self.assertRaises(TypeError):
            encoder.default("not a numpy array")


if __name__ == '__main__':
    unittest.main()