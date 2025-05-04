import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_252 import custom_parse_query

query_string = "param1=value1&param2="

import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    parsed_values = custom_parse_query(query_string)
    if w:
        for warn in w:
            assert not issubclass(
                warn.category, DeprecationWarning
            ), "Deprecated API used!"

expect1 = "value1"
expect2 = ""
assert parsed_values.get("param1") == expect1
assert parsed_values.get("param2") == expect2
