# Add the parent directory to import sys
import os
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from falcon.stream import BoundedStream
from sample_244 import custom_writable

import io
import warnings

stream = io.BytesIO(b"initial data")
bstream = BoundedStream(stream, 1024)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    writable_val = custom_writable(bstream)
    if w:
        for warn in w:
            assert not issubclass(
                warn.category, DeprecationWarning
            ), "Deprecated API used!"

expect = False
assert writable_val == expect
