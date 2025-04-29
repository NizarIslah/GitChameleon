import os
import sys
import unittest

import falcon
from falcon import testing

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sample_254 import handle_error


class TestResource:
    def on_get(self, req, resp):
        """Raise an exception to trigger the error handler."""
        raise Exception('Test exception')


class TestHandleError(unittest.TestCase):
    def setUp(self):
        # Create a Falcon App instance with our error handler
        self.app = falcon.App()
        self.app.add_error_handler(Exception, handle_error)
        self.app.add_route('/test_error', TestResource())
        self.client = testing.TestClient(self.app)

    def test_handle_error_response(self):
        """Test that the error handler properly formats the response."""
        result = self.client.simulate_get('/test_error')
        self.assertEqual(result.status, falcon.HTTP_500)
        response_data = result.json
        self.assertIn('error', response_data)
        self.assertEqual(response_data['error'], 'Test exception')
        self.assertIn('details', response_data)
        self.assertEqual(response_data['details']['request'], '/test_error')
        self.assertIsInstance(response_data['details']['params'], dict)


if __name__ == '__main__':
    unittest.main()
