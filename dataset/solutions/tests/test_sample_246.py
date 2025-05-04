import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_246 import custom_environ


class TestCustomEnviron(unittest.TestCase):
    def test_custom_environ_returns_dict(self):
        """Test that custom_environ returns a dictionary."""
        result = custom_environ("1.1")
        self.assertIsInstance(result, dict)

    def test_custom_environ_sets_http_version(self):
        """Test that custom_environ sets the HTTP version correctly."""
        # Test with HTTP 1.0
        result_1_0 = custom_environ("1.0")
        self.assertEqual(result_1_0.get("SERVER_PROTOCOL"), "HTTP/1.0")

        # Test with HTTP 1.1
        result_1_1 = custom_environ("1.1")
        self.assertEqual(result_1_1.get("SERVER_PROTOCOL"), "HTTP/1.1")

        # Test with HTTP 2.0
        result_2_0 = custom_environ("2.0")
        self.assertEqual(result_2_0.get("SERVER_PROTOCOL"), "HTTP/2")


if __name__ == "__main__":
    unittest.main()
