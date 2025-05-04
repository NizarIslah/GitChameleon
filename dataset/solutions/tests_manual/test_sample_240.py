import os
import sys
import unittest

from falcon import Response

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_240 import custom_body_length

info = "Falcon"


class DummyResponse(Response):
    pass


resp = DummyResponse()

import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    custom_resp = custom_body_length(resp, info)
    if w:
        for warn in w:
            assert not issubclass(
                warn.category, DeprecationWarning
            ), "Deprecated API used!"
expect = str(len(info))
assert custom_resp.content_length == expect
