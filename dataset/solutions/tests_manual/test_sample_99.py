import unittest
from django.utils import timezone
from datetime import timezone as py_timezone
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_99 import get_time_in_utc


year = 2024
month = 11
day = 5
utc_time = get_time_in_utc(year, month, day)
assertion_value = utc_time.tzname() == "UTC"
assert assertion_value
assertion_value = utc_time.isoformat() == "2024-11-05T00:00:00+00:00"
assert assertion_value
