import json

# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np
from scipy.stats import hmean
import flask  # <-- Added import

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_172 import MyCustomJSONHandler, app, data, eval


class TestSample172(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.testing = True
        self.client = self.app.test_client()

    def test_data_route(self):
        """Test the /data route with a valid numpy array"""
        with self.app.test_request_context():
            # Test with a simple array
            test_array = np.array([[1, 2, 3], [4, 5, 6]])
            response = data(test_array)

            # Check response type
            self.assertIsInstance(response, flask.Response)

            # Parse the JSON response
            response_data = json.loads(response.get_data(as_text=True))

            # Check the structure of the response
            self.assertIn("numbers", response_data)

            # Expected result: harmonic means of each row
            expected = [hmean(np.array([1, 2, 3])), hmean(np.array([4, 5, 6]))]
            self.assertEqual(response_data["numbers"], expected)

    def test_eval_function(self):
        """Test the eval function with a valid numpy array"""
        test_array = np.array([[1, 2, 3], [4, 5, 6]])
        result = eval(app, data, test_array)

        # Parse the binary JSON response
        response_data = json.loads(result.decode("utf-8"))

        # Check the structure of the response
        self.assertIn("numbers", response_data)

        # Expected result: harmonic means of each row
        expected = [hmean(np.array([1, 2, 3])), hmean(np.array([4, 5, 6]))]
        self.assertEqual(response_data["numbers"], expected)

    def test_custom_json_encoder_with_numpy_array(self):
        """Test the custom JSON encoder with a numpy array"""
        encoder = MyCustomJSONHandler()

        # Test with a simple array
        test_array = np.array([[1, 2, 3], [4, 5, 6]])
        encoded = encoder.default(test_array)

        # Expected result: harmonic means of each row
        expected = [hmean(np.array([1, 2, 3])), hmean(np.array([4, 5, 6]))]
        self.assertEqual(encoded, expected)

    def test_custom_json_encoder_with_nan_values(self):
        """Test the custom JSON encoder with NaN values in the array"""
        encoder = MyCustomJSONHandler()

        # Test with an array containing NaN
        test_array = np.array([[1, 2, np.nan], [4, 5, 6]])
        encoded = encoder.default(test_array)

        # Expected result: NaN for the first row, harmonic mean for the second
        self.assertTrue(np.isnan(encoded[0]))
        self.assertEqual(encoded[1], hmean(np.array([4, 5, 6])))

    def test_custom_json_encoder_with_non_numpy_object(self):
        """Test the custom JSON encoder with a non-numpy object"""
        encoder = MyCustomJSONHandler()

        # This should raise a TypeError as the default method doesn't handle strings
        with self.assertRaises(TypeError):
            encoder.default("not a numpy array")


if __name__ == "__main__":
    unittest.main()
