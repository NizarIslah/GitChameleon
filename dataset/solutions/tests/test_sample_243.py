# Add the parent directory to import sys
import os
import sys
import unittest
from typing import Any, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_243 import custom_environ


class TestCustomEnviron(unittest.TestCase):
    """Test cases for the custom_environ function in sample_243.py."""

    def test_custom_environ_returns_dict(self):
        """Test that custom_environ returns a dictionary."""
        result = custom_environ("test_path")
        self.assertIsInstance(result, dict)

    def test_custom_environ_sets_root_path(self):
        """Test that custom_environ sets the root_path correctly."""
        test_path = "/test/path"
        result = custom_environ(test_path)

        # In Falcon 3.0.0, the root_path is set in the SCRIPT_NAME environment variable
        self.assertEqual(result.get("SCRIPT_NAME"), test_path)

    def test_custom_environ_with_empty_string(self):
        """Test custom_environ with an empty string."""
        result = custom_environ("")
        self.assertEqual(result.get("SCRIPT_NAME"), "")

    def test_custom_environ_default_values(self):
        """Test that custom_environ includes default values from falcon.testing.create_environ."""
        result = custom_environ("test_path")

        # Check for some standard environment variables that should be present
        self.assertIn("REQUEST_METHOD", result)
        self.assertIn("wsgi.input", result)
        self.assertIn("wsgi.errors", result)


if __name__ == "__main__":
    unittest.main()
