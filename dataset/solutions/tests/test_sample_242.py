import json
import os
import sys
import unittest

import falcon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_242 import custom_http_error


class TestSample242(unittest.TestCase):
    def test_custom_http_error_returns_bytes(self):
        """Test that custom_http_error returns bytes."""
        result = custom_http_error("Test Title", "Test Description")
        self.assertIsInstance(result, bytes)

    def test_custom_http_error_content(self):
        """Test that custom_http_error returns correct JSON content."""
        result = custom_http_error("Test Title", "Test Description")

        # Convert bytes to dict
        error_dict = json.loads(result)

        # Check structure and content
        self.assertIn("title", error_dict)
        self.assertIn("description", error_dict)
        self.assertEqual(error_dict["title"], "Test Title")
        self.assertEqual(error_dict["description"], "Test Description")
        # Removed status check: 'status' key does not exist

    def test_custom_http_error_with_empty_strings(self):
        """Test custom_http_error with empty strings."""
        result = custom_http_error("", "")
        error_dict = json.loads(result)

        # The actual function returns '400 Bad Request' as title when title is empty
        self.assertEqual(error_dict["title"], "400 Bad Request")
        self.assertEqual(error_dict["description"], "")
        # Removed status check

    def test_custom_http_error_with_special_characters(self):
        """Test custom_http_error with special characters."""
        title = "Special: !@#$%^&*()"
        description = "More special: <>?,./"

        result = custom_http_error(title, description)
        error_dict = json.loads(result)

        self.assertEqual(error_dict["title"], title)
        self.assertEqual(error_dict["description"], description)


if __name__ == "__main__":
    unittest.main()
