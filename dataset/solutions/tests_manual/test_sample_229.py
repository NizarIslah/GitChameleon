import pytest
import pathlib
import sys
from unittest.mock import MagicMock

# Add the parent directory to sys.path to import the module
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Import the module to test
import sample_229


import inspect


def test_pytest_collect_file_signature():
    sig = inspect.signature(pytest_collect_file)
    params = list(sig.parameters.items())
    name, param = params[0]
    expect = pathlib.Path
    assert param.annotation == expect


test_pytest_collect_file_signature()
