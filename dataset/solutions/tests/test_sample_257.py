import unittest
import sys
import os

# Add the parent directory to import sys
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sample_257 import CustomRouter, solution

import falcon


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
        self.assertIn('GET', stored_method_map)
        self.assertIn('POST', stored_method_map)
        self.assertEqual(method_map, stored_method_map)

    def test_add_route_with_fallback(self):
        """Test that add_route correctly handles the fallback parameter."""
        # Create a resource with only one method
        class LimitedResource:
            def on_get(self, req, resp):
                pass

        # Create a fallback handler
        def fallback_handler(req, resp):
            pass

        resource = LimitedResource()
        uri_template = "/limited"

        # Add the route with a fallback
        method_map = self.router.add_route(uri_template, resource, fallback=fallback_handler)

        # Verify the route was added with fallback
        self.assertIn(uri_template, self.router.routes)
        _, stored_method_map = self.router.routes[uri_template]
        
        # Check that GET uses the resource method
        self.assertEqual(stored_method_map['GET'].__self__, resource)
        
        # Check that other methods like POST use the fallback
        self.assertEqual(stored_method_map['POST'], fallback_handler)


if __name__ == '__main__':
    unittest.main()