import os
import sys
import unittest
from unittest.mock import MagicMock

import falcon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_245 import ExampleMiddleware, custom_middleware_variable

import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    middleware = custom_middleware_variable()
    prepared_mw = app_helpers.prepare_middleware(middleware)
    if w:
        for warn in w:
            assert not issubclass(
                warn.category, DeprecationWarning
            ), "Deprecated API used!"

expect = (list, tuple)
assert isinstance(prepared_mw, expect)
