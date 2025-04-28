import unittest
import json
import numpy as np
from flask import Flask
import sys
import os

# Add the parent directory to import sys
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_163 import app, data, eval, MyCustomJSONHandler

class TestSample163(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_data_route(self):
        """Test the /data route with various numpy arrays"""
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
        """Test the /data route with arrays containing NaN values"""
        # Test with array containing NaN values
        test_arr = np.array([1.0, 2.0, np.nan, 3.0, np.nan])
        with self.app.test_request_context():
            response = data(test_arr)
            response_data = json.loads(response.get_data(as_text=True))
            # Check length (should be 4: 1.0, 2.0, 3.0, and one NaN)
            self.assertEqual(len(response_data['numbers']), 4)
            # Check that the first three elements are the unique non-NaN values
            self.assertEqual(response_data['numbers'][:3], [1.0, 2.0, 3.0])
            # The last element should be null (NaN in JSON)
            self.assertIsNone(response_data['numbers'][3])
    
    def test_eval_function(self):
        """Test the eval function"""
        test_arr = np.array([1, 2, 3, 4, 5])
        result = eval(self.app, data, test_arr)
        # Convert bytes to string and then to JSON
        result_json = json.loads(result.decode('utf-8'))
        self.assertEqual(result_json['numbers'], [1, 2, 3, 4, 5])
    
    def test_custom_json_encoder(self):
        """Test the custom JSON encoder directly"""
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
        # Check length (should be 4: 1.0, 2.0, 3.0, and one NaN)
        self.assertEqual(len(encoded), 4)
        # Check that the first three elements are the unique non-NaN values
        self.assertEqual(encoded[:3], [1.0, 2.0, 3.0])
        # The last element should be NaN
        self.assertTrue(np.isnan(encoded[3]))
        
        # Test with non-numpy object (should raise TypeError)
        with self.assertRaises(TypeError):
            encoder.default("not a numpy array")

if __name__ == '__main__':
    unittest.main()