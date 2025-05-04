# Add the parent directory to import sys
import os
import sys
import unittest
from typing import Any, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_243 import custom_environ


info = "/my/root/path"

import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    env = custom_environ(info)
    if w:
        for warn in w:
            assert not issubclass(
                warn.category, DeprecationWarning
            ), "Deprecated API used!"
expect = info
assert env.get("SCRIPT_NAME", "") == expect
