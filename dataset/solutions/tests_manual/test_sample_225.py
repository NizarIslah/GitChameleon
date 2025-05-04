import os

# Add the parent directory to the path so we can import the sample module
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mitmproxy.http import Headers
from sample_225 import custom_function


header_name = b"Content-Type"
initial_value = b"text/html"

expect = "text/html"
results = custom_function(header_name, initial_value)
assert results.get(header_name) == expect
