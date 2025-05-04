import os
import sys
import unittest

import falcon
from falcon import testing

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_238 import custom_body

resp = falcon.Response()
info = "Falcon"

import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    resp = custom_body(resp, info)
    if w:
        assert issubclass(
            w[-1].category, DeprecationWarning
        ), "Expected a DeprecationWarning but got something else!"

expect = "Falcon"
assert resp.text == expect
