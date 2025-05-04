import os
import sys
import unittest
from unittest.mock import MagicMock

import falcon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_249 import custom_link

link_rel = "next"
link_href = "http://example.com/next"
resp = Response()

import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    custom_resp = custom_link(resp, link_rel, link_href)
    if w:
        for warn in w:
            assert not issubclass(
                warn.category, DeprecationWarning
            ), "Deprecated API used!"

expected_link = f"<{link_href}>;"
link_header = custom_resp.get_header("Link") or ""
assert expected_link in link_header
