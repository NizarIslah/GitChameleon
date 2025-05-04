import datetime

# Add the parent directory to import sys
import os
import sys
import unittest

from flask import helpers

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_151 import convert_timedelta_to_seconds


import datetime

td = datetime.timedelta(hours=2, minutes=30, microseconds=1)
assertion_results = convert_timedelta_to_seconds(td) == 9000
assert assertion_results
