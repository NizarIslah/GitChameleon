import os

# Add the directory containing sample_237.py to the Python path
import sys
import unittest
from io import BytesIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_237 import DummyRequest, get_bounded_stream


class TestSample237(unittest.TestCase):
    def test_dummy_request_initialization(self):
        """Test that DummyRequest initializes correctly with the provided data."""
        test_data = b"test data"
        req = DummyRequest(test_data)

        # Check that the stream is a BytesIO instance
        self.assertIsInstance(req.stream, BytesIO)

        # Check that content_length is set correctly
        self.assertEqual(req.content_length, len(test_data))

        # Check that the stream contains the expected data
        self.assertEqual(req.stream.getvalue(), test_data)

    def test_bounded_stream_read(self):
        """Test that the bounded stream can be read and contains the expected data."""
        test_data = b"falcon stream test"
        req = DummyRequest(test_data)

        bounded_stream = get_bounded_stream(req)

        # Read the data from the bounded stream
        read_data = bounded_stream.read()

        # Check that the read data matches the original data
        self.assertEqual(read_data, test_data)

        # After reading, the stream should be at the end
        self.assertEqual(bounded_stream.read(), b"")

    def test_bounded_stream_partial_read(self):
        """Test that the bounded stream can be partially read."""
        test_data = b"partial read test"
        req = DummyRequest(test_data)

        bounded_stream = get_bounded_stream(req)

        # Read part of the data
        partial_data = bounded_stream.read(7)
        self.assertEqual(partial_data, b"partial")

        # Read the rest of the data
        remaining_data = bounded_stream.read()
        self.assertEqual(remaining_data, b" read test")


if __name__ == "__main__":
    unittest.main()
