import os

# Add the parent directory to import sys
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import falcon
from sample_257 import CustomRouter, solution


class DummyResource:
    def on_get(self, req, resp):
        resp.text = "hello"


import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    router = CustomRouter()
    solution()
    method_map = router.add_route("/test", DummyResource())
    if w:
        for warn in w:
            assert not issubclass(
                warn.category, DeprecationWarning
            ), "Deprecated API used!"

expect = "/test"
assert expect in router.routes
resource, mapping = router.routes["/test"]
assert callable(mapping.get("GET", None))
