import os
import sys
import unittest

import falcon
from falcon import testing

# Ensure we can import from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sample_254 import handle_error


class TestHandleError(unittest.TestCase):
    def setUp(self):
        # Create a Falcon App instance with our error handler
        self.app = falcon.App()
        self.app.add_error_handler(Exception, handle_error)
        
        # Create a test client
        self.client = testing.TestClient(self.app)
        
        # Add a test route that raises an exception
        self.app.add_route('/test_error', TestResource())
    
    def test_handle_error_response(self):
        """Test that the error handler properly formats the response."""
        # Make a request to the test endpoint that will raise an exception
        result = self.client.simulate_get('/test_error')
        
        # Check status code
        self.assertEqual(result.status, falcon.HTTP_500)
        
        # Check response body structure
        response_data = result.json
        self.assertIn('error', response_data)
        self.assertEqual(response_data['error'], 'Test exception')
        
        # Check details
        self.assertIn('details', response_data)
        self.assertEqual(response_data['details']['request'], '/test_error')
        self.assertIsInstance(response_data['details']['params'], dict)
    
    def test_unknown_path(self):
        """Test handling when request path is not available."""
        # Create a resource that will trigger the error handler with a modified request
        self.app.add_route('/no_path', NoPathResource())
        
        # Make a request
        result = self.client.simulate_get('/no_path')
        
        # Check response
        self.assertEqual(result.status, falcon.HTTP_500)
        response_data = result.json
        self.assertEqual(response_data['details']['request'], 'unknown')


# Test resources
class TestResource:
    def on_get(self, req, resp):
        """Raise an exception to trigger the error handler."""
        raise Exception('Test exception')


class NoPathResource:
    def on_get(self, req, resp):
        """Remove path attribute and raise exception."""
        delattr(req, 'path')
        raise Exception('No path exception')


if __name__ == '__main__':
    unittest.main()