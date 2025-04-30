import unittest
from django.test import TestCase

# Import the module to test
from sample_105 import Square, create_square, display_side_and_area


class TestSquareModel(TestCase):
    """
    Converted to Django's built-in TestCase so that the test database and tables
    are set up automatically. This removes the need for manual schema editing.
    """

    def test_square_creation(self):
        """Test that a Square instance is created correctly with the right side value."""
        square = Square.objects.create(side=5)
        self.assertEqual(square.side, 5)
        self.assertEqual(square.area, 25)  # 5*5 = 25

    def test_create_square_function(self):
        """Test the create_square function."""
        square = create_square(7)
        self.assertEqual(square.side, 7)
        self.assertEqual(square.area, 49)  # 7*7 = 49

        # Verify it's saved in the database
        db_square = Square.objects.get(id=square.id)
        self.assertEqual(db_square.side, 7)
        self.assertEqual(db_square.area, 49)

    def test_display_side_and_area(self):
        """Test the display_side_and_area function."""
        square = create_square(10)
        side, area = display_side_and_area(square)
        self.assertEqual(side, 10)
        self.assertEqual(area, 100)  # 10*10 = 100

    def test_area_calculation(self):
        """Test that the area is calculated correctly when saving."""
        # Test with different side values
        test_cases = [3, 10, 15, 100]

        for side in test_cases:
            square = create_square(side)
            self.assertEqual(square.area, side * side)

    def test_area_update(self):
        """Test that the area is updated when the side is changed."""
        square = create_square(5)
        self.assertEqual(square.area, 25)

        # Update the side
        square.side = 8
        square.save()

        # Refresh from database to ensure the change was saved
        square.refresh_from_db()
        self.assertEqual(square.side, 8)
        self.assertEqual(square.area, 64)  # 8*8 = 64


if __name__ == '__main__':
    unittest.main()