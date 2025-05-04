# Add the parent directory to import sys
import os
import sys
import unittest

from django.utils import timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_100 import get_time_in_utc


class TestGetTimeInUTC(unittest.TestCase):
    def test_get_time_in_utc_returns_correct_datetime(self):
        """Test that get_time_in_utc returns a datetime with the correct values and UTC timezone."""
        # Arrange
        year, month, day = 2023, 5, 15

        # Act
        result = get_time_in_utc(year, month, day)

        # Assert
        self.assertEqual(result.year, year)
        self.assertEqual(result.month, month)
        self.assertEqual(result.day, day)
        self.assertEqual(result.hour, 0)
        self.assertEqual(result.minute, 0)
        self.assertEqual(result.second, 0)
        self.assertEqual(result.tzinfo, timezone.utc)

    def test_get_time_in_utc_timezone_aware(self):
        """Test that the returned datetime is timezone-aware with UTC timezone."""
        # Act
        result = get_time_in_utc(2023, 5, 15)

        # Assert
        self.assertTrue(timezone.is_aware(result))
        self.assertEqual(str(result.tzinfo), "UTC")


if __name__ == "__main__":
    unittest.main()
