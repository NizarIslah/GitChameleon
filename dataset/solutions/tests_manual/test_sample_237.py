import os

# Add the directory containing sample_237.py to the Python path
import sys
import unittest
from io import BytesIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_237 import DummyRequest, get_bounded_stream

test_data = b"Hello, Falcon!"
req = DummyRequest(test_data)

bounded_stream = get_bounded_stream(req)
read_data = bounded_stream.read()
expect = b"Hello, Falcon!"
assert read_data == expect
