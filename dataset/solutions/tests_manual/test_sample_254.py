import sys
import unittest

import falcon
from falcon.testing import TestClient
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_254 import handle_error


class DummyReq:
    pass


class DummyResp:
    def __init__(self):
        self.media = None
        self.status = None


dummy_req = DummyReq()
dummy_resp = DummyResp()
dummy_ex = Exception("Test error")
dummy_params = {}

import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    handle_error(dummy_req, dummy_resp, dummy_ex, dummy_params)
    if w:
        for warn in w:
            assert not issubclass(
                warn.category, DeprecationWarning
            ), "Deprecated API used!"
expect1 = {"error": "Test error", "details": {"request": "unknown", "params": {}}}
assert dummy_resp.media == expect1

expect2 = falcon.HTTP_500
assert dummy_resp.media == expect1
assert dummy_resp.status == expect2
