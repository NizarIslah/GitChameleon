import os

# Add the parent directory to the path so we can import the sample module
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mitmproxy.http import Headers
from sample_225 import custom_function


class TestSample225(unittest.TestCase):
    def test_custom_function_creates_headers(self):
        # Test with basic header
        header_name = b"Content-Type"
        initial_value = b"application/json"

        result = custom_function(header_name, initial_value)

        # Verify the result is a Headers object
        self.assertIsInstance(result, Headers)

        # Verify the header was set correctly
        self.assertEqual(result[header_name], initial_value.decode("utf-8"))

    def test_custom_function_with_empty_values(self):
        # Test with empty header value
        header_name = b"X-Empty"
        initial_value = b""

        result = custom_function(header_name, initial_value)

        self.assertIsInstance(result, Headers)
        self.assertEqual(result[header_name], initial_value.decode("utf-8"))

    def test_custom_function_with_multiple_headers(self):
        # The function only adds one header, but we can verify the Headers object
        # behaves as expected when we access a non-existent header
        header_name = b"X-Test"
        initial_value = b"test-value"

        result = custom_function(header_name, initial_value)

        # Verify the header we added exists
        self.assertEqual(result[header_name], initial_value.decode("utf-8"))

        # Verify a non-existent header returns None
        self.assertIsNone(result.get(b"X-NonExistent"))

    def test_custom_function_preserves_bytes(self):
        # Test with non-ASCII bytes to ensure they're preserved
        header_name = b"X-Special"
        initial_value = b"\xe2\x98\x83"  # Snowman emoji in UTF-8

        result = custom_function(header_name, initial_value)

        self.assertEqual(result[header_name], initial_value.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
