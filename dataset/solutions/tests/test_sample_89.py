# Test for sample_89.py
import unittest
import ctypes
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_89 import c_str


class TestCSrConversion(unittest.TestCase):
    """Test cases for the c_str function in sample_89.py."""

    def test_c_str_returns_c_char_p(self):
        """Test that c_str returns a ctypes.c_char_p object."""
        result = c_str("test string")
        self.assertIsInstance(result, ctypes.c_char_p)

    def test_c_str_content(self):
        """Test that the content of the returned c_char_p matches the input string."""
        test_string = "Hello, world!"
        result = c_str(test_string)
        # Convert the c_char_p back to a Python string for comparison
        # In Python 3, c_char_p.value returns bytes, so we decode it
        self.assertEqual(result.value.decode("utf-8"), test_string)

    def test_c_str_empty_string(self):
        """Test that c_str handles empty strings correctly."""
        result = c_str("")
        self.assertEqual(result.value, b"")

    def test_c_str_special_characters(self):
        """Test that c_str handles strings with special characters."""
        test_string = "Special chars: !@#$%^&*()_+{}[]|\\:;\"'<>,.?/~`"
        result = c_str(test_string)
        self.assertEqual(result.value.decode("utf-8"), test_string)

    def test_c_str_unicode(self):
        """Test that c_str handles Unicode characters."""
        test_string = "Unicode: 你好, こんにちは, 안녕하세요"
        result = c_str(test_string)
        self.assertEqual(result.value.decode("utf-8"), test_string)


if __name__ == "__main__":
    unittest.main()
