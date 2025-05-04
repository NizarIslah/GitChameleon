import unittest
import time
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_216 import custom_client
import mitmproxy.connection as conn

ip_address = "127.0.0.1"
i_port = 111
o_port = 222
output_client = custom_client(ip_address, i_port, o_port)

expect_peername = ("127.0.0.1", 111)
expect_sockname = ("127.0.0.1", 222)

assert output_client.peername == expect_peername
assert output_client.sockname == expect_sockname
