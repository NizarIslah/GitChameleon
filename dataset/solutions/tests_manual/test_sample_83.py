import unittest
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the function to test
from sample_83 import decode_string

ENCODED_STRING = b"\x68\x65\x6c\x6c\x6f"
expected = "hello"
assert decode_string(ENCODED_STRING) == expected
