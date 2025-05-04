import unittest
import sys
import io
from contextlib import redirect_stdout
from unittest.mock import patch
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_218 import DummyServerConn, ConnectionLogger, solution


class TestSample218(unittest.TestCase):
    def setUp(self):
        # Reset ConnectionLogger class before each test
        ConnectionLogger.server_connected = None

    def test_dummy_server_conn_initialization(self):
        """Test that DummyServerConn initializes with the correct sockname."""
        sockname = ("127.0.0.1", 8080)
        server_conn = DummyServerConn(sockname)
        self.assertEqual(server_conn.sockname, sockname)

    def test_solution_adds_server_connected_method(self):
        """Test that solution() adds server_connected method to ConnectionLogger."""
        # Verify server_connected is not defined before calling solution
        self.assertFalse(
            hasattr(ConnectionLogger, "server_connected")
            and callable(ConnectionLogger.server_connected)
        )

        # Call solution to add the method
        solution()

        # Verify server_connected is now defined
        self.assertTrue(
            hasattr(ConnectionLogger, "server_connected")
            and callable(ConnectionLogger.server_connected)
        )

    def test_server_connected_prints_sockname(self):
        """Test that server_connected method prints the sockname."""
        # Call solution to add the method
        solution()

        # Create a server connection with a test sockname
        sockname = ("192.168.1.1", 443)
        server_conn = DummyServerConn(sockname)

        # Capture stdout to verify the print
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            # Call the method on an instance of ConnectionLogger
            logger = ConnectionLogger()
            logger.server_connected(server_conn)

        # Verify the output
        self.assertEqual(captured_output.getvalue().strip(), str(sockname))


if __name__ == "__main__":
    unittest.main()
