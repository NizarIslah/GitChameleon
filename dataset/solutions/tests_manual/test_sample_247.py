import os
import sys
import unittest

import falcon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_247 import custom_append_link

resp = Response()
link = "http://example.com"
rel = "preconnect"

response = custom_append_link(resp, link, rel)
expected = "crossorigin"
assert expected in response.get_header("Link")
