# Add the parent directory to import sys
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_248 import custom_falcons


class TestSample248(unittest.TestCase):
    """Test cases for sample_248.py which uses the Falcon web framework."""

    def test_custom_falcons_returns_falcon_app(self):
        """Test that custom_falcons() returns a Falcon App instance."""
        app = custom_falcons()

        # Verify that the returned object is a Falcon App instance
        import falcon

        self.assertIsInstance(app, falcon.App)

    def test_falcon_app_properties(self):
        """Test that the Falcon App has expected properties and behaviors."""
        app = custom_falcons()

        # Check that the app has the expected attributes of a Falcon App
        self.assertTrue(hasattr(app, "add_route"))
        self.assertTrue(hasattr(app, "add_middleware"))
        self.assertTrue(hasattr(app, "add_sink"))

        # Verify the app can handle basic operations like adding a route
        class DummyResource:
            def on_get(self, req, resp):
                resp.body = "Hello, World!"

        # This should not raise any exceptions
        app.add_route("/hello", DummyResource())


if __name__ == "__main__":
    unittest.main()
