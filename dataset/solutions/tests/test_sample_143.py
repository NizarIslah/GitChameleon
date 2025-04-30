import json
# Add the parent directory to import sys
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sample_143


class TestSample143(unittest.TestCase):
    def setUp(self):
        # Create a test Flask app
        self.app = sample_143.flask.Flask('test')
        # Apply the custom JSON encoder
        sample_143.app_set_up(self.app)
        # Configure the app for testing
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Register the data route with the test app
        @self.app.route('/data')
        def test_data(num_set=None):
            return sample_143.flask.jsonify({'numbers': num_set})
    
    def test_json_encoder_with_set(self):
        """Test that the custom JSON encoder correctly handles sets."""
        with self.app.app_context():
            # Create a set of numbers
            test_set = {3, 1, 2, 5, 4}
            # Use Flask's jsonify which should use our custom encoder
            response = sample_143.flask.jsonify({'numbers': test_set})
            # Get the JSON data
            data = json.loads(response.get_data(as_text=True))
            # Check that the set was converted to a sorted list
            self.assertEqual(data['numbers'], [1, 2, 3, 4, 5])
    
    def test_data_route_with_set(self):
        """Test the data route with a set parameter."""
        # Create a test set
        test_set = {3, 1, 2, 5, 4}
        
        # Use the eval function from the module to test the route
        result = sample_143.eval(self.app, test_data, test_set)
        
        # Convert bytes to string and load as JSON
        result_str = result.decode('utf-8')
        data = json.loads(result_str)
        
        # Check that the set was converted to a sorted list
        self.assertEqual(data['numbers'], [1, 2, 3, 4, 5])
    
    def test_app_set_up(self):
        """Test that app_set_up correctly sets the JSON encoder."""
        # Create a new Flask app
        test_app = sample_143.flask.Flask('test_setup')
        
        # Initially, the app should use the default JSON encoder
        self.assertNotEqual(test_app.json_encoder, sample_143.MyCustomJSONHandler)
        
        # Apply our custom setup
        sample_143.app_set_up(test_app)
        
        # Now the app should use our custom JSON encoder
        self.assertEqual(test_app.json_encoder, sample_143.MyCustomJSONHandler)

if __name__ == '__main__':
    unittest.main()