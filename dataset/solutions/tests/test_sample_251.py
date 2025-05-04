import os
import sys
import unittest

import falcon
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_251 import raise_too_large_error


class TestSample251(unittest.TestCase):
    """Test cases for sample_251.py functions."""

    def test_raise_too_large_error(self):
        """Test that raise_too_large_error raises the correct exception with the provided message."""
        error_message = "Payload too large"

        # Verify that the function raises the expected exception with the correct message
        with pytest.raises(falcon.HTTPPayloadTooLarge) as excinfo:
            raise_too_large_error(error_message)

        # Check that the exception contains the error message
        assert str(excinfo.value) == error_message

    def test_raise_too_large_error_empty_message(self):
        """Test that raise_too_large_error works with an empty message."""
        error_message = ""

        with pytest.raises(falcon.HTTPPayloadTooLarge) as excinfo:
            raise_too_large_error(error_message)

        assert str(excinfo.value) == error_message

    def test_raise_too_large_error_long_message(self):
        """Test that raise_too_large_error works with a long message."""
        error_message = "This is a very long error message " * 10

        with pytest.raises(falcon.HTTPPayloadTooLarge) as excinfo:
            raise_too_large_error(error_message)

        assert str(excinfo.value) == error_message


if __name__ == "__main__":
    unittest.main()
