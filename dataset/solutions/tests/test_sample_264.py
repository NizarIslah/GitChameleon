# Add the parent directory to import sys
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_264 import DummyConnection, custom_write, tornado


class TestSample264(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a new HTTPServerRequest for each test
        self.request = tornado.httputil.HTTPServerRequest(method="GET", uri="/")
        self.request.connection = DummyConnection()

    def test_dummy_connection_initialization(self):
        """Test that DummyConnection initializes with an empty buffer."""
        connection = DummyConnection()
        self.assertEqual(connection.buffer, [])

    def test_dummy_connection_write(self):
        """Test that DummyConnection.write adds chunks to the buffer."""
        connection = DummyConnection()
        connection.write(b"Hello")
        self.assertEqual(connection.buffer, [b"Hello"])

        # Test multiple writes
        connection.write(b" World")
        self.assertEqual(connection.buffer, [b"Hello", b" World"])

    def test_custom_write_returns_buffer(self):
        """Test that custom_write returns the connection buffer."""
        result = custom_write(self.request, b"Test Message")
        self.assertEqual(result, [b"Test Message"])

    def test_custom_write_appends_to_buffer(self):
        """Test that custom_write appends to an existing buffer."""
        # First write
        custom_write(self.request, b"First")
        # Second write
        result = custom_write(self.request, b"Second")
        # Check that both writes are in the buffer
        self.assertEqual(result, [b"First", b"Second"])

    def test_custom_write_with_string_input(self):
        """Test custom_write with string input (should handle bytes conversion)."""
        # The function accepts str but tornado might expect bytes
        # This test verifies the function works with string input
        result = custom_write(self.request, "String Input")
        self.assertIn("String Input", str(result))

    def test_custom_write_with_empty_string(self):
        """Test custom_write with an empty string."""
        result = custom_write(self.request, "")
        self.assertEqual(result, [""])


if __name__ == "__main__":
    unittest.main()
