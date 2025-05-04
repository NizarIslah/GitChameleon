import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_246 import custom_environ

import warnings

version = "1.1"
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    env = custom_environ(version)
    if w:
        for warn in w:
            assert not issubclass(
                warn.category, DeprecationWarning
            ), "Deprecated API used!"

expect = "HTTP/1.1"
assert env.get("SERVER_PROTOCOL", "") == expect
