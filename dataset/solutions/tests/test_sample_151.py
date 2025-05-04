import datetime

# Add the parent directory to import sys
import os
import sys
import unittest

from flask import helpers

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_151 import convert_timedelta_to_seconds


class TestTimedeltaConversion(unittest.TestCase):
    def test_convert_timedelta_to_seconds(self):
        # Test with various timedelta values

        # Test with 1 second
        td = datetime.timedelta(seconds=1)
        self.assertEqual(convert_timedelta_to_seconds(td), 1)

        # Test with 1 minute
        td = datetime.timedelta(minutes=1)
        self.assertEqual(convert_timedelta_to_seconds(td), 60)

        # Test with 1 hour
        td = datetime.timedelta(hours=1)
        self.assertEqual(convert_timedelta_to_seconds(td), 3600)

        # Test with 1 day
        td = datetime.timedelta(days=1)
        self.assertEqual(convert_timedelta_to_seconds(td), 86400)

        # Test with a combination of units
        td = datetime.timedelta(days=1, hours=2, minutes=3, seconds=4)
        expected_seconds = 86400 + 7200 + 180 + 4  # 93784
        self.assertEqual(convert_timedelta_to_seconds(td), expected_seconds)

        # Test with zero
        td = datetime.timedelta()
        self.assertEqual(convert_timedelta_to_seconds(td), 0)

        # Test with microseconds
        td = datetime.timedelta(microseconds=1000000)  # 1 second
        self.assertEqual(convert_timedelta_to_seconds(td), 1)


if __name__ == "__main__":
    unittest.main()
