import unittest
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the function to test
from sample_83 import decode_string


class TestSample83(unittest.TestCase):
    """Test cases for the decode_string function in sample_83.py."""

    def test_decode_string_ascii(self):
        """Test decoding a simple ASCII string."""
        input_bytes = b"hello world"
        expected = "hello world"
        result = decode_string(input_bytes)
        self.assertEqual(result, expected)
        self.assertIsInstance(result, str)

    def test_decode_string_utf8(self):
        """Test decoding a UTF-8 encoded string with non-ASCII characters."""
        input_bytes = b"caf\xc3\xa9"  # 'café' in UTF-8
        expected = "café"
        result = decode_string(input_bytes)
        self.assertEqual(result, expected)
        self.assertIsInstance(result, str)

    def test_decode_string_empty(self):
        """Test decoding an empty bytes object."""
        input_bytes = b""
        expected = ""
        result = decode_string(input_bytes)
        self.assertEqual(result, expected)
        self.assertIsInstance(result, str)

    def test_decode_string_special_chars(self):
        """Test decoding bytes with special characters."""
        input_bytes = b"\xe2\x82\xac\xf0\x9f\x98\x80"  # '€😀' in UTF-8
        expected = "€😀"
        result = decode_string(input_bytes)
        self.assertEqual(result, expected)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
