import unittest
from django.utils import timezone
from datetime import timezone as py_timezone
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_99 import get_time_in_utc


class TestGetTimeInUTC(unittest.TestCase):
    def test_get_time_in_utc_returns_datetime_with_utc_timezone(self):
        # Arrange
        year, month, day = 2023, 12, 31

        # Act
        result = get_time_in_utc(year, month, day)

        # Assert
        self.assertIsInstance(result, timezone.datetime)
        self.assertEqual(result.year, year)
        self.assertEqual(result.month, month)
        self.assertEqual(result.day, day)
        self.assertEqual(result.tzinfo, py_timezone.utc)

    def test_get_time_in_utc_with_different_dates(self):
        # Test with a few different dates
        test_cases = [
            (2020, 1, 1),
            (2022, 6, 15),
            (2024, 2, 29),  # Leap year
        ]

        for year, month, day in test_cases:
            with self.subTest(year=year, month=month, day=day):
                result = get_time_in_utc(year, month, day)

                self.assertEqual(result.year, year)
                self.assertEqual(result.month, month)
                self.assertEqual(result.day, day)
                self.assertEqual(result.tzinfo, py_timezone.utc)

    def test_get_time_in_utc_has_zero_time_components(self):
        # Verify that hours, minutes, seconds are all zero
        result = get_time_in_utc(2023, 1, 1)

        self.assertEqual(result.hour, 0)
        self.assertEqual(result.minute, 0)
        self.assertEqual(result.second, 0)
        self.assertEqual(result.microsecond, 0)


if __name__ == "__main__":
    unittest.main()
