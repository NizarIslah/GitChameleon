import os
import sys
import unittest

import falcon
from falcon import testing

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_238 import custom_body


class TestCustomBody(unittest.TestCase):
    def setUp(self):
        # Create a Falcon app for testing
        self.app = falcon.App()

        # Create a test resource that uses the custom_body function
        class TestResource:
            def on_get(self, req, resp):
                custom_body(resp, "Test message")

        self.app.add_route("/test", TestResource())
        self.client = testing.TestClient(self.app)

    def test_custom_body_sets_text(self):
        # Test that the function sets the response text correctly
        resp = falcon.Response()
        test_message = "Hello, world!"
        result = custom_body(resp, test_message)

        # Check that the text was set correctly
        self.assertEqual(resp.text, test_message)

        # Check that the function returns the response object
        self.assertIs(result, resp)

    def test_custom_body_in_request_context(self):
        # Test the function in a real request context
        result = self.client.simulate_get("/test")

        # Check that the response has the expected text
        self.assertEqual(result.text, "Test message")

        # Check that the response status is 200 OK (default)
        self.assertEqual(result.status, falcon.HTTP_200)


if __name__ == "__main__":
    unittest.main()
