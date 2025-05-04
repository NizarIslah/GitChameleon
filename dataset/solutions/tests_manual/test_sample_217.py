import unittest
from unittest import mock
import os
import sys
import mitmproxy.connection as conn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_217 import custom_server

ip_address = "192.168.1.1"
server_port = 80
output_server = custom_server(ip_address, server_port)
expect = ("192.168.1.1", 80)
assert output_server.address == expect
