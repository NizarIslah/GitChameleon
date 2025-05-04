import unittest
import io
import sys
from contextlib import redirect_stdout
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from sample_221 import DummyClientConn, ConnectionLogger, solution


class TestSample221(unittest.TestCase):
    def test_dummy_client_conn_initialization(self):
        """Test that DummyClientConn correctly stores the peername."""
        peername = ("127.0.0.1", 8080)
        client_conn = DummyClientConn(peername)
        self.assertEqual(client_conn.peername, peername)

    def test_solution_adds_client_connected_method(self):
        """Test that solution() adds the client_connected method to ConnectionLogger."""
        solution()
        self.assertTrue(hasattr(ConnectionLogger, "client_connected"))

    def test_client_connected_prints_peername(self):
        """Test that the client_connected method prints the peername."""
        # Call solution to add the method
        solution()

        # Create a test client connection
        peername = ("192.168.1.1", 12345)
        client_conn = DummyClientConn(peername)

        # Capture stdout
        f = io.StringIO()
        with redirect_stdout(f):
            # Create an instance of ConnectionLogger and call client_connected
            logger = ConnectionLogger()
            logger.client_connected(client_conn)

        # Get the printed output
        output = f.getvalue().strip()

        # Check that the peername was printed
        self.assertEqual(output, str(peername))


if __name__ == "__main__":
    unittest.main()
