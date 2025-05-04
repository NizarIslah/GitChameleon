import datetime

# Add the parent directory to import sys
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sample_152 import convert_timedelta_to_seconds


class TestTimedeltaConversion(unittest.TestCase):
    def test_convert_timedelta_to_seconds(self):
        # Test with zero timedelta
        zero_td = datetime.timedelta(seconds=0)
        self.assertEqual(convert_timedelta_to_seconds(zero_td), 0)

        # Test with positive seconds
        seconds_td = datetime.timedelta(seconds=42)
        self.assertEqual(convert_timedelta_to_seconds(seconds_td), 42)

        # Test with minutes and seconds
        minutes_td = datetime.timedelta(minutes=5, seconds=30)
        self.assertEqual(
            convert_timedelta_to_seconds(minutes_td), 330
        )  # 5*60 + 30 = 330

        # Test with hours, minutes, and seconds
        hours_td = datetime.timedelta(hours=2, minutes=30, seconds=15)
        self.assertEqual(
            convert_timedelta_to_seconds(hours_td), 9015
        )  # 2*3600 + 30*60 + 15 = 9015

        # Test with days and other units
        days_td = datetime.timedelta(days=1, hours=12, minutes=30, seconds=45)
        self.assertEqual(
            convert_timedelta_to_seconds(days_td), 131445
        )  # 1*86400 + 12*3600 + 30*60 + 45 = 131445

        # Test with negative timedelta
        negative_td = datetime.timedelta(seconds=-60)
        self.assertEqual(convert_timedelta_to_seconds(negative_td), -60)


if __name__ == "__main__":
    unittest.main()
