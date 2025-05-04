import unittest
import sys
import io
from unittest.mock import patch
from contextlib import redirect_stdout

# Import the module to test
sys.path.append("/repo/dataset/solutions")
from sample_222 import DummyClientConn, ConnectionLogger, solution


class TestSample222(unittest.TestCase):
    def setUp(self):
        # Reset ConnectionLogger class before each test
        ConnectionLogger.client_disconnected = None

    def test_dummy_client_conn_init(self):
        """Test that DummyClientConn initializes with the correct peername."""
        peername = ("127.0.0.1", 8080)
        client_conn = DummyClientConn(peername)
        self.assertEqual(client_conn.peername, peername)

    def test_solution_adds_method(self):
        """Test that solution() adds client_disconnected method to ConnectionLogger."""
        # Verify method doesn't exist before calling solution
        self.assertFalse(
            hasattr(ConnectionLogger, "client_disconnected")
            and callable(getattr(ConnectionLogger, "client_disconnected"))
        )

        # Call solution
        solution()

        # Verify method exists after calling solution
        self.assertTrue(
            hasattr(ConnectionLogger, "client_disconnected")
            and callable(getattr(ConnectionLogger, "client_disconnected"))
        )

    def test_client_disconnected_output(self):
        """Test that client_disconnected prints the peername."""
        # Call solution to add the method
        solution()

        # Create a test instance and client connection
        logger = ConnectionLogger()
        peername = ("192.168.1.1", 12345)
        client_conn = DummyClientConn(peername)

        # Capture stdout
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            logger.client_disconnected(client_conn)

        # Check output
        self.assertEqual(captured_output.getvalue().strip(), str(peername))


if __name__ == "__main__":
    unittest.main()
