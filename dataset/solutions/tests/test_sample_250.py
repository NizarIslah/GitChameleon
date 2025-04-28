import json
# Import the function to test
import os
import sys
import unittest

from falcon import Request
from falcon.testing import create_environ

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sample_250 import custom_media


class TestSample250(unittest.TestCase):
    def test_custom_media_json(self):
        """Test that custom_media correctly returns JSON media from a request."""
        # Create test data
        test_data = {"key": "value", "number": 42}
        
        # Create a request with JSON media
        headers = {"Content-Type": "application/json"}
        body = json.dumps(test_data).encode()
        env = create_environ(
            method='POST',
            path='/',
            headers=headers,
            body=body
        )
        req = Request(env)
        
        # Call the function under test
        result = custom_media(req)
        
        # Assert the result matches our test data
        self.assertEqual(result, test_data)
    
    def test_custom_media_empty(self):
        """Test that custom_media handles empty request body."""
        # Create a request with empty body
        env = create_environ(
            method='POST',
            path='/',
            headers={"Content-Type": "application/json"},
            body=b''
        )
        req = Request(env)
        
        # Call the function under test
        result = custom_media(req)
        
        # Assert the result is an empty dict
        self.assertEqual(result, {})


if __name__ == '__main__':
    unittest.main()