import unittest
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.solutions.sample_248 import custom_falcons


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
        self.assertTrue(hasattr(app, 'add_route'))
        self.assertTrue(hasattr(app, 'add_middleware'))
        self.assertTrue(hasattr(app, 'add_sink'))
        
        # Verify the app can handle basic operations like adding a route
        class DummyResource:
            def on_get(self, req, resp):
                resp.body = 'Hello, World!'
                
        # This should not raise any exceptions
        app.add_route('/hello', DummyResource())


if __name__ == '__main__':
    unittest.main()