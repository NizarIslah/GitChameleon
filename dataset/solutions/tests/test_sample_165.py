import json

# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_165 import MyCustomJSONHandler, app, data, eval


class TestSample165(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_data_route(self):
        """Test the /data route with a test client"""
        # Create a test array
        test_array = np.array([1, 2, 3, 4])

        # Test the route directly using the test client
        with self.app.test_request_context():
            response = data(test_array)
            self.assertEqual(response.status_code, 200)

            # Parse the response data
            response_data = json.loads(response.get_data(as_text=True))
            self.assertIn("numbers", response_data)
            self.assertEqual(response_data["numbers"], [1, 2, 3, 4])

    def test_eval_function(self):
        """Test the eval function"""
        # Create a test array
        test_array = np.array([1, 2, 3, 4])

        # Call the eval function
        result = eval(self.app, data, test_array)

        # Parse the result
        result_data = json.loads(result)
        self.assertIn("numbers", result_data)
        self.assertEqual(result_data["numbers"], [1, 2, 3, 4])

    def test_custom_json_encoder_with_ndarray(self):
        """Test the custom JSON encoder with numpy arrays"""
        # Create a test array
        test_array = np.array([[1, 2], [3, 4]])

        # Create an instance of the custom encoder
        encoder = MyCustomJSONHandler()

        # Test the encoder directly
        encoded = encoder.default(test_array)
        self.assertEqual(encoded, [1, 3, 2, 4])  # Transposed and flattened

        # Test JSON serialization with the custom encoder
        json_str = json.dumps({"data": test_array}, cls=MyCustomJSONHandler)
        decoded = json.loads(json_str)
        self.assertEqual(decoded["data"], [1, 3, 2, 4])

    def test_custom_json_encoder_with_non_ndarray(self):
        """Test the custom JSON encoder with non-numpy objects"""
        # Test with a regular list
        test_list = [1, 2, 3, 4]

        # Create an instance of the custom encoder
        encoder = MyCustomJSONHandler()

        # This should raise TypeError as default() should be called for unsupported types
        with self.assertRaises(TypeError):
            encoder.default(test_list)

        # But JSON serialization should work fine
        json_str = json.dumps({"data": test_list}, cls=MyCustomJSONHandler)
        decoded = json.loads(json_str)
        self.assertEqual(decoded["data"], [1, 2, 3, 4])

    def test_2d_array_handling(self):
        """Test handling of 2D arrays"""
        # Create a 2D test array
        test_array = np.array([[1, 2, 3], [4, 5, 6]])

        # Test with the eval function
        result = eval(self.app, data, test_array)

        # Parse the result
        result_data = json.loads(result)
        self.assertIn("numbers", result_data)

        # The array should be transposed and flattened
        self.assertEqual(result_data["numbers"], [1, 4, 2, 5, 3, 6])


if __name__ == "__main__":
    unittest.main()
