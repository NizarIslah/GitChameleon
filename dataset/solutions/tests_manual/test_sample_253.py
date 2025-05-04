import unittest
from unittest.mock import MagicMock
from falcon import Request
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_253 import custom_get_param

import warnings
from falcon.testing import create_environ
import json

json_value = json.dumps({"bar": "baz"})
query_string = f"foo={json_value}"

env = create_environ(query_string=query_string)
req = Request(env)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = custom_get_param(req)
    if w:
        for warn in w:
            assert not issubclass(
                warn.category, DeprecationWarning
            ), "Deprecated API used!"

expect = {"bar": "baz"}
assert result == expect
