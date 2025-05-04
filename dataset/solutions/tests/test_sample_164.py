import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_164 import MyCustomJSONHandler, app, data, eval_app


class TestSample164(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_data_route(self):
        """Test the /data route with various numpy arrays"""
        # Test with regular numpy array
        test_array = np.array([1, 2, 3, 2, 1])
        with self.app.test_request_context():
            response = data(test_array)
            result = json.loads(response.get_data(as_text=True))
            self.assertEqual(result["numbers"], [1, 2, 3])

    def test_data_route_with_duplicates(self):
        """Test that the custom JSON handler removes duplicates"""
        test_array = np.array([5, 5, 5, 10, 10, 15])
        with self.app.test_request_context():
            response = data(test_array)
            result = json.loads(response.get_data(as_text=True))
            self.assertEqual(result["numbers"], [5, 10, 15])

    def test_data_route_with_nan(self):
        """Test handling of NaN values in numpy arrays"""
        test_array = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
        with self.app.test_request_context():
            response = data(test_array)
            result = json.loads(response.get_data(as_text=True))
            # np.unique with equal_nan=False treats each NaN as unique, so we expect 5 unique values
            self.assertEqual(len(result["numbers"]), 5)  # 1.0, 2.0, 3.0, nan, nan

            # Check that 1.0, 2.0, 3.0 are in the result
            self.assertIn(1.0, result["numbers"])
            self.assertIn(2.0, result["numbers"])
            self.assertIn(3.0, result["numbers"])

            # Check that there are two NaN values
            nan_count = sum(np.isnan(x) for x in result["numbers"])
            self.assertEqual(nan_count, 2)

    def test_eval_app_function(self):
        """Test the eval_app function"""
        test_array = np.array([7, 7, 8, 9, 9])
        result = eval_app(self.app, data, test_array)
        result_dict = json.loads(result)
        self.assertEqual(result_dict["numbers"], [7, 8, 9])

    def test_custom_json_handler(self):
        """Test the MyCustomJSONHandler directly"""
        handler = MyCustomJSONHandler(self.app)

        # Test with numpy array
        test_array = np.array([4, 4, 5, 6, 6])
        result = handler.default(test_array)
        self.assertEqual(result, [4, 5, 6])

        # Test with non-numpy object (should raise TypeError as per default implementation)
        class CustomObject:
            pass

        with self.assertRaises(TypeError):
            handler.default(CustomObject())


if __name__ == "__main__":
    unittest.main()
