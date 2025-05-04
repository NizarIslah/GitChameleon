import os
import sys
import unittest
import io

from falcon import Request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_256 import custom_set_context


class TestCustomSetContext(unittest.TestCase):
    def setUp(self):
        # Create a minimal valid WSGI environment for Falcon
        self.env = {
            "REQUEST_METHOD": "GET",
            "PATH_INFO": "/",
            "QUERY_STRING": "",
            "wsgi.input": io.BytesIO(b""),
            "wsgi.errors": io.StringIO(),
            "SERVER_NAME": "localhost",
            "SERVER_PORT": "80",
            "SERVER_PROTOCOL": "HTTP/1.1",
        }
        self.req = Request(self.env)

    def test_custom_set_context_sets_values(self):
        # Test that the function sets the role and user correctly
        role = "admin"
        user = "john_doe"

        context = custom_set_context(self.req, role, user)

        # Verify that the context has the correct values
        self.assertEqual(context.role, role)
        self.assertEqual(context.user, user)

        # Also verify that the request's context was updated
        self.assertEqual(self.req.context.role, role)
        self.assertEqual(self.req.context.user, user)

    def test_custom_set_context_returns_context(self):
        # Test that the function returns the context object
        context = custom_set_context(self.req, "user", "jane_doe")

        # Verify that the returned object is the request's context
        self.assertIs(context, self.req.context)

    def test_custom_set_context_with_empty_values(self):
        # Test with empty strings
        role = ""
        user = ""

        context = custom_set_context(self.req, role, user)

        # Verify that empty strings are set correctly
        self.assertEqual(context.role, role)
        self.assertEqual(context.user, user)


if __name__ == "__main__":
    unittest.main()
