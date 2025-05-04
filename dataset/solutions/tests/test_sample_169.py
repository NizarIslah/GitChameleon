import json

# Add the parent directory to import sys
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_169 import MyCustomJSONHandler, app, data, eval_app


class TestSample169(unittest.TestCase):
    """Test cases for the Flask app and JSON handler in sample_169.py."""

    def test_data_route_with_list(self):
        """Test the data route with a regular list."""
        test_list = [1, 2, 3, 4, 5]
        with app.test_request_context():
            response = data(test_list)
            response_data = json.loads(response.get_data(as_text=True))

            self.assertEqual(response_data, {"numbers": test_list})
            self.assertEqual(response.status_code, 200)

    def test_eval_app_function(self):
        """Test the eval_app function."""
        test_list = [10, 20, 30]
        result = eval_app(app, data, test_list)

        # Parse the JSON result
        result_data = json.loads(result)

        self.assertEqual(result_data, {"numbers": test_list})

    def test_custom_json_handler_with_regular_object(self):
        """Test the custom JSON handler with a regular object."""
        test_dict = {"key": "value"}

        # Create an instance of the custom JSON handler
        handler = MyCustomJSONHandler(app)

        # The handler should delegate to the default implementation for regular objects
        result = json.dumps(test_dict, default=handler.default)
        self.assertEqual(json.loads(result), test_dict)

    def test_custom_json_handler_with_1d_array(self):
        """Test the custom JSON handler with a 1D numpy array."""
        test_array = np.array([1, 2, 3])

        # Create an instance of the custom JSON handler
        handler = MyCustomJSONHandler(app)

        # This should use the default numpy array serialization
        with self.assertRaises(TypeError):
            # Default JSON serializer can't handle numpy arrays
            json.dumps(test_array, default=handler.default)

    def test_custom_json_handler_with_2d_array(self):
        """Test the custom JSON handler with a 2D numpy array."""
        test_array = np.array([[1, 2], [3, 4]])

        # Create an instance of the custom JSON handler
        handler = MyCustomJSONHandler(app)

        # This should use the default numpy array serialization
        with self.assertRaises(TypeError):
            # Default JSON serializer can't handle numpy arrays
            json.dumps(test_array, default=handler.default)

    def test_custom_json_handler_with_3d_square_array(self):
        """Test the custom JSON handler with a 3D numpy array with square matrices."""
        # Create a 3D array with shape (2, 3, 3) - last two dimensions are equal
        test_array = np.array(
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[9, 8, 7], [6, 5, 4], [3, 2, 1]]]
        )

        # Create an instance of the custom JSON handler
        handler = MyCustomJSONHandler(app)

        # Calculate expected determinants
        expected_det_0 = np.linalg.det(test_array[0])
        expected_det_1 = np.linalg.det(test_array[1])
        expected = [expected_det_0, expected_det_1]

        # Serialize with our custom handler
        result = json.dumps(test_array, default=handler.default)
        result_data = json.loads(result)

        # Check that the result is a list of determinants
        self.assertIsInstance(result_data, list)
        self.assertEqual(len(result_data), 2)

        # Check that the determinants are correct (with floating point tolerance)
        np.testing.assert_almost_equal(result_data, expected)

    def test_custom_json_handler_with_3d_non_square_array(self):
        """Test the custom JSON handler with a 3D numpy array with non-square matrices."""
        # Create a 3D array with shape (2, 2, 3) - last two dimensions are not equal
        test_array = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

        # Create an instance of the custom JSON handler
        handler = MyCustomJSONHandler(app)

        # This should use the default numpy array serialization
        with self.assertRaises(TypeError):
            # Default JSON serializer can't handle numpy arrays
            json.dumps(test_array, default=handler.default)

    def test_app_integration_with_3d_square_array(self):
        """Test the integration of the app with a 3D square array."""
        # Create a 3D array with shape (2, 2, 2) - last two dimensions are equal
        test_array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

        # Calculate expected determinants
        expected_det_0 = np.linalg.det(test_array[0])  # det([[1, 2], [3, 4]])
        expected_det_1 = np.linalg.det(test_array[1])  # det([[5, 6], [7, 8]])
        expected = [expected_det_0, expected_det_1]

        # Use the eval_app function to test the integration
        result = eval_app(app, data, test_array)
        result_data = json.loads(result)

        # Check that the result contains the numbers key with the determinants
        self.assertIn("numbers", result_data)
        np.testing.assert_almost_equal(result_data["numbers"], expected)


if __name__ == "__main__":
    unittest.main()
