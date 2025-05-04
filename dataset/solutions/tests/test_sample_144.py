import json

# Add the parent directory to import sys
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_144 import app, app_set_up, data, eval


class TestSample144(unittest.TestCase):
    def setUp(self):
        # Configure the app for testing
        app.config["TESTING"] = True
        app.config["DEBUG"] = False
        # Set up the custom JSON provider
        app_set_up(app)
        self.client = app.test_client()

    def test_data_function(self):
        """Test the data function directly"""
        # Create a test set
        test_set = {1, 3, 2, 5, 4}

        # Use the eval function to test the data function
        result = eval(app, data, test_set)

        # Parse the result and verify
        result_json = json.loads(result)
        self.assertIn("numbers", result_json)
        # The set should be converted to a sorted list
        self.assertEqual(result_json["numbers"], [1, 2, 3, 4, 5])

    def test_custom_json_handler(self):
        """Test that the custom JSON handler correctly serializes sets"""
        # Create a test set
        test_set = {3, 1, 4, 2}

        # Test with app context
        with app.app_context():
            # Use Flask's jsonify which should use our custom handler
            result = app.json.dumps({"data": test_set})
            result_dict = json.loads(result)

            # Verify the set was serialized as a sorted list
            self.assertEqual(result_dict["data"], [1, 2, 3, 4])

    def test_eval_function(self):
        """Test the eval helper function"""

        # Create a simple function that returns JSON
        def test_fn(data):
            return app.json.response({"result": data})

        # Test with different data types
        test_data = {1, 2, 3}

        # Use the eval function
        with app.app_context():
            result = eval(app, test_fn, test_data)
            result_dict = json.loads(result)

            # Verify the result
            self.assertEqual(result_dict["result"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
