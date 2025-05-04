import pytest
import pathlib
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sample_232


import inspect


def test_pytest_report_collectionfinish_signature():
    sig = inspect.signature(pytest_report_collectionfinish)
    params = list(sig.parameters.items())
    name, param = params[0]
    expect = pathlib.Path
    assert param.annotation == expect


test_pytest_report_collectionfinish_signature()
