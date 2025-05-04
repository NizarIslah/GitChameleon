import pytest
import pathlib
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sample_232


def test_pytest_report_collectionfinish_hook_exists():
    """Test that the pytest_report_collectionfinish hook exists in the module."""
    assert hasattr(sample_232, "pytest_report_collectionfinish")
    assert callable(sample_232.pytest_report_collectionfinish)


def test_pytest_report_collectionfinish_accepts_path_parameter():
    """Test that the hook accepts a pathlib.Path parameter."""
    # Create a mock Path object
    mock_path = pathlib.Path(".")

    # Call the function with the mock path
    # This should not raise any exceptions if the parameter type is correct
    sample_232.pytest_report_collectionfinish(mock_path)
