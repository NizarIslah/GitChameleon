import json

# Import the function to test
import os
import sys
import unittest

from falcon import Request
from falcon.testing import create_environ

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_250 import custom_media

import warnings

payload = {"key": "value"}
body_bytes = json.dumps(payload).encode("utf-8")

env = create_environ(body=body_bytes, headers={"Content-Type": "application/json"})

req = Request(env)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    media = custom_media(req)
    if w:
        for warn in w:
            assert not issubclass(
                warn.category, DeprecationWarning
            ), "Deprecated API used!"
expect = payload
assert media == expect
