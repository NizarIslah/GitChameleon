import unittest
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to import the module
sys.path.append(str(Path(__file__).parent.parent))
from sample_224 import generate_cert_new


def test_generate_cert_new():
    hostname = "example.com"
    cert_pem, key_pem = generate_cert_new(hostname)

    assert hostname in cert_pem, "Hostname not found in certificate PEM"

    assert cert_pem.strip() != "", "Certificate PEM is empty"
    assert key_pem.strip() != "", "Key PEM is empty"


test_generate_cert_new()
