import unittest
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to import the module
sys.path.append(str(Path(__file__).parent.parent))
from sample_224 import generate_cert_new


class TestGenerateCertNew(unittest.TestCase):
    def test_generate_cert_new_returns_tuple(self):
        """Test that generate_cert_new returns a tuple."""
        result = generate_cert_new("example.com")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_generate_cert_new_returns_strings(self):
        """Test that generate_cert_new returns two strings."""
        cert_pem, key_pem = generate_cert_new("example.com")
        self.assertIsInstance(cert_pem, str)
        self.assertIsInstance(key_pem, str)

    def test_generate_cert_new_includes_hostname(self):
        """Test that the generated certificate and key include the hostname."""
        hostname = "test.example.com"
        cert_pem, key_pem = generate_cert_new(hostname)
        self.assertIn(hostname, cert_pem)
        self.assertIn(hostname, key_pem)

    def test_generate_cert_new_different_hostnames(self):
        """Test that different hostnames produce different certificates."""
        cert1, key1 = generate_cert_new("site1.example.com")
        cert2, key2 = generate_cert_new("site2.example.com")

        self.assertNotEqual(cert1, cert2)
        self.assertNotEqual(key1, key2)


if __name__ == "__main__":
    unittest.main()
