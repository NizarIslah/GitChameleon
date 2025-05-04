import unittest
from unittest import mock
import os
import sys
import mitmproxy.connection as conn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_217 import custom_server


class TestCustomServer(unittest.TestCase):
    def test_custom_server_creation(self):
        """Test that custom_server creates a Server object with the correct address."""
        # Test parameters
        ip_address = "192.168.1.1"
        server_port = 8080

        # Call the function
        server = custom_server(ip_address, server_port)

        # Verify the result
        self.assertIsInstance(server, conn.Server)
        self.assertEqual(server.address, (ip_address, server_port))

    def test_custom_server_with_different_values(self):
        """Test custom_server with different IP and port values."""
        # Test with different parameters
        ip_address = "127.0.0.1"
        server_port = 443

        # Call the function
        server = custom_server(ip_address, server_port)

        # Verify the result
        self.assertEqual(server.address, (ip_address, server_port))


if __name__ == "__main__":
    unittest.main()
