import json
import os
import sys
import unittest

import falcon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_242 import custom_http_error

title = "Bad Request"
description = "An error occurred"
import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = custom_http_error(title, description)
    if w:
        for warn in w:
            assert not issubclass(
                warn.category, DeprecationWarning
            ), "Deprecated API used!"


expect = b'{"title": "Bad Request", "description": "An error occurred"}'
assert result == expect
