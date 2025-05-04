import os
import sys
import unittest
import io

from falcon import Request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_256 import custom_set_context

from falcon.testing import create_environ

env = create_environ()
req = Request(env)
role = "trial"
user = "guest"
import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    context = custom_set_context(req, role, user)
    if w:
        for warn in w:
            assert not issubclass(
                warn.category, DeprecationWarning
            ), "Deprecated API used!"

expect1 = "trial"
expect2 = "guest"

assert context.role == expect1
assert context.user == expect2
