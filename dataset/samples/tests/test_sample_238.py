import unittest
import falcon
from falcon.testing import SimpleTestClient
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.solutions.sample_238 import custom_body


class TestCustomBody(unittest.TestCase):
    def setUp(self):
        # Create a Falcon app for testing
        self.app = falcon.App()
        self.client = SimpleTestClient(self.app)
        
        # Create a test resource that uses the custom_body function
        class TestResource:
            def on_get(self, req, resp):
                custom_body(resp, "Test message")
                
        self.app.add_route('/test', TestResource())

    def test_custom_body_sets_text(self):
        # Test that the function sets the response text correctly
        resp = falcon.Response()
        test_message = "Hello, world!"
        result = custom_body(resp, test_message)
        
        # Check that the text was set correctly
        self.assertEqual(resp.text, test_message)
        
        # Check that the function returns the response object
        self.assertEqual(result, resp)
        
    def test_custom_body_in_request_context(self):
        # Test the function in a real request context
        result = self.client.simulate_get('/test')
        
        # Check that the response has the expected text
        self.assertEqual(result.text, "Test message")
        
        # Check that the response status is 200 OK (default)
        self.assertEqual(result.status, falcon.HTTP_200)


if __name__ == '__main__':
    unittest.main()