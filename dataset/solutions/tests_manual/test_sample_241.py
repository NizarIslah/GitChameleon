import os
import sys
import unittest
from unittest.mock import MagicMock

import falcon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_241 import custom_data


class DummyResponse(Response):
    pass


info = "Falcon data"

resp = DummyResponse()
import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    rendered_body = custom_data(resp, info)
    if w:
        for warn in w:
            assert not issubclass(
                warn.category, DeprecationWarning
            ), "Deprecated API used!"


expect = info
assert rendered_body == expect
