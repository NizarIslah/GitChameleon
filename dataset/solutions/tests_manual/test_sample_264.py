# Add the parent directory to import sys
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_264 import DummyConnection, custom_write, tornado

written_data = custom_write(req, "Hello, Tornado!")
expect = ["Hello, Tornado!"]
assert written_data == expect
