import unittest
import time
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_216 import custom_client
import mitmproxy.connection as conn


class TestSample216(unittest.TestCase):
    """Test cases for sample_216.py which contains a custom_client function."""

    def test_custom_client_returns_client_object(self):
        """Test that custom_client returns a Client object."""
        # Arrange
        ip_address = "127.0.0.1"
        i_port = 8080
        o_port = 9090

        # Act
        result = custom_client(ip_address, i_port, o_port)

        # Assert
        self.assertIsInstance(result, conn.Client)

    def test_custom_client_sets_correct_peername(self):
        """Test that custom_client sets the correct peername."""
        # Arrange
        ip_address = "192.168.1.1"
        i_port = 8080
        o_port = 9090

        # Act
        result = custom_client(ip_address, i_port, o_port)

        # Assert
        self.assertEqual(result.peername, (ip_address, i_port))

    def test_custom_client_sets_correct_sockname(self):
        """Test that custom_client sets the correct sockname."""
        # Arrange
        ip_address = "10.0.0.1"
        i_port = 8080
        o_port = 9090

        # Act
        result = custom_client(ip_address, i_port, o_port)

        # Assert
        self.assertEqual(result.sockname, (ip_address, o_port))

    @patch("time.time")
    def test_custom_client_sets_timestamp(self, mock_time):
        """Test that custom_client sets the timestamp_start correctly."""
        # Arrange
        mock_time.return_value = 12345.6789
        ip_address = "127.0.0.1"
        i_port = 8080
        o_port = 9090

        # Act
        result = custom_client(ip_address, i_port, o_port)

        # Assert
        self.assertEqual(result.timestamp_start, 12345.6789)

    def test_custom_client_with_different_parameters(self):
        """Test that custom_client works with different parameters."""
        # Arrange
        test_cases = [
            ("127.0.0.1", 80, 8080),
            ("192.168.0.1", 443, 4443),
            ("10.0.0.1", 8000, 9000),
            ("0.0.0.0", 1, 2),
        ]

        for ip, i_port, o_port in test_cases:
            # Act
            result = custom_client(ip, i_port, o_port)

            # Assert
            self.assertEqual(result.peername, (ip, i_port))
            self.assertEqual(result.sockname, (ip, o_port))
            self.assertIsInstance(result.timestamp_start, float)


if __name__ == "__main__":
    unittest.main()
