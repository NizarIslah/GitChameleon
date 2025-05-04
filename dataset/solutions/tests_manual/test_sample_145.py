# Add the parent directory to import sys
import os
import sys
import unittest
from io import BytesIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_145 import app1, download, get_content_disp


content_disp = get_content_disp(app1, download)
assertion_result = "filename=hello.txt" in content_disp
assert assertion_result
