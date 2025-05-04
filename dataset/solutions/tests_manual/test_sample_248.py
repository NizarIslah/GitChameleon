# Add the parent directory to import sys
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_248 import custom_falcons


class TestSample248(unittest.TestCase):
    """Test cases for sample_248.py which uses the Falcon web framework."""

    def test_custom_falcons_returns_falcon_app(self):
        """Test that custom_falcons() returns a Falcon App instance."""
        app = custom_falcons()

        # Verify that the returned object is a Falcon App instance
        import falcon


import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    app_instance = custom_falcons()
    if w:
        for warn in w:
            assert not issubclass(
                warn.category, DeprecationWarning
            ), "Deprecated API used!"

expect = falcon.App
assert isinstance(app_instance, expect)
