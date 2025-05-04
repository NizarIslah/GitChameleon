import json

# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_166 import MyCustomJSONHandler, app, data, eval_app


class TestSample166(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_data_route_with_list(self):
        """Test the data route with a regular list"""
        test_list = [1, 2, 3, 4, 5]

        # Test using the function directly
        with self.app.test_request_context():
            response = data(test_list)
            result = json.loads(response.get_data(as_text=True))
            self.assertEqual(result, {"numbers": test_list})

    def test_data_route_with_numpy_array(self):
        """Test the data route with a numpy array"""
        # Create a 2D numpy array
        test_array = np.array([[1, 2, 3], [4, 5, 6]])

        # Test using the eval_app function
        result = eval_app(self.app, data, test_array)
        result_dict = json.loads(result)

        # The custom JSON handler should transpose and flatten the array
        expected = test_array.T.copy().flatten().tolist()
        self.assertEqual(result_dict, {"numbers": expected})

    def test_custom_json_handler(self):
        """Test the custom JSON handler directly"""
        # Create a numpy array
        test_array = np.array([[1, 2], [3, 4]])

        # Create an instance of the custom JSON handler
        handler = MyCustomJSONHandler(self.app)

        # Test the default method
        result = handler.default(test_array)

        # Expected: transposed, flattened array
        expected = test_array.T.copy().flatten().tolist()
        self.assertEqual(result, expected)

    def test_custom_json_handler_non_array(self):
        """Test the custom JSON handler with non-array objects"""

        # Create a custom object that the default handler can't handle
        class CustomObject:
            pass

        test_obj = CustomObject()

        # Create an instance of the custom JSON handler
        handler = MyCustomJSONHandler(self.app)

        # The handler should raise TypeError for objects it can't handle
        with self.assertRaises(TypeError):
            handler.default(test_obj)

    def test_eval_app_function(self):
        """Test the eval_app function"""
        test_list = [10, 20, 30]

        # Use eval_app to test the data function
        result = eval_app(self.app, data, test_list)
        result_dict = json.loads(result)

        self.assertEqual(result_dict, {"numbers": test_list})


if __name__ == "__main__":
    unittest.main()
