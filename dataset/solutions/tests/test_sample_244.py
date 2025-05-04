# Add the parent directory to import sys
import os
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from falcon.stream import BoundedStream
from sample_244 import custom_writable


class TestCustomWritable(unittest.TestCase):
    def test_custom_writable_returns_true(self):
        # Create a mock BoundedStream that returns True for writable()
        mock_stream = Mock(spec=BoundedStream)
        mock_stream.writable.return_value = True

        # Test that custom_writable returns True when the stream is writable
        self.assertTrue(custom_writable(mock_stream))
        # Verify that writable() was called
        mock_stream.writable.assert_called_once()

    def test_custom_writable_returns_false(self):
        # Create a mock BoundedStream that returns False for writable()
        mock_stream = Mock(spec=BoundedStream)
        mock_stream.writable.return_value = False

        # Test that custom_writable returns False when the stream is not writable
        self.assertFalse(custom_writable(mock_stream))
        # Verify that writable() was called
        mock_stream.writable.assert_called_once()


if __name__ == "__main__":
    unittest.main()
