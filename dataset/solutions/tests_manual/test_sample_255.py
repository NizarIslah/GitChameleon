# Import the function to test
import os
import sys
import unittest
from unittest.mock import MagicMock

from falcon import Request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_255 import custom_get_dpr

from falcon.testing import create_environ

env = create_environ(query_string="dpr=2")
req = Request(env)

import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    dpr = custom_get_dpr(req)
    if w:
        for warn in w:
            assert not issubclass(
                warn.category, DeprecationWarning
            ), "Deprecated API used!"

expect = 2
assert dpr == expect
