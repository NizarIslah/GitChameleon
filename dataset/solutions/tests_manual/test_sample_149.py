# Add the parent directory to import sys
import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_149 import error404, safe_join_fail_404


base_path = "/var/www/myapp"
sub_path = "../secret.txt"

try:
    joined = safe_join_fail_404(base_path, sub_path)
except werkzeug.exceptions.NotFound as e:
    assertion_result = True
else:
    assertion_result = False
assert assertion_result

base_path = "/var/www/myapp"
sub_path = "secret.txt"
joined = safe_join_fail_404(base_path, sub_path)
assertion_result = joined == "/var/www/myapp/secret.txt"
assert assertion_result
