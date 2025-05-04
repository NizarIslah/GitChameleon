import os

# Add the parent directory to import sys
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import falcon
from sample_257 import CustomRouter, solution


class TestCustomRouter(unittest.TestCase):
    def setUp(self):
        # Call the solution function to add the add_route method to CustomRouter
        solution()
        self.router = CustomRouter()

    def test_init(self):
        """Test that the CustomRouter initializes with an empty routes dictionary."""
        self.assertEqual({}, self.router.routes)

    def test_add_route(self):
        """Test that add_route correctly adds a route to the router."""

        # Create a simple resource class with HTTP method handlers
        class TestResource:
            def on_get(self, req, resp):
                pass

            def on_post(self, req, resp):
                pass

        resource = TestResource()
        uri_template = "/test"

        # Add the route
        method_map = self.router.add_route(uri_template, resource)

        # Verify the route was added correctly
        self.assertIn(uri_template, self.router.routes)
        stored_resource, stored_method_map = self.router.routes[uri_template]

        # Check that the resource is stored correctly
        self.assertEqual(resource, stored_resource)

        # Check that the method map contains the expected methods
        self.assertIn("GET", stored_method_map)
        self.assertIn("POST", stored_method_map)
        self.assertEqual(method_map, stored_method_map)


if __name__ == "__main__":
    unittest.main()
