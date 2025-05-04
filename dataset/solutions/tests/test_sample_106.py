# Add the parent directory to import sys
import os
import sys
import unittest
from django.db import connection
from django.db.utils import OperationalError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_106 import Square, create_square, display_side_and_area


class TestSquareModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Create the necessary tables in the in-memory database
        with connection.schema_editor() as schema_editor:
            schema_editor.create_model(Square)

    @classmethod
    def tearDownClass(cls):
        # Clean up after all tests
        with connection.schema_editor() as schema_editor:
            schema_editor.delete_model(Square)

    def setUp(self):
        # Clean up the database before each test
        Square.objects.all().delete()

    def test_create_square(self):
        # Test creating a square with side=5
        square = create_square(5)

        # Verify the square was created with correct side
        self.assertEqual(square.side, 5)

        # Verify the area was calculated correctly
        self.assertEqual(square.area, 25)

        # Verify the object was saved to the database
        self.assertIsNotNone(square.id)

        # Verify we can retrieve it from the database
        retrieved_square = Square.objects.get(id=square.id)
        self.assertEqual(retrieved_square.side, 5)
        self.assertEqual(retrieved_square.area, 25)

    def test_display_side_and_area(self):
        # Create a square to test with
        square = create_square(7)

        # Test the display function
        side, area = display_side_and_area(square)

        # Verify correct values are returned
        self.assertEqual(side, 7)
        self.assertEqual(area, 49)

    def test_multiple_squares(self):
        # Test creating multiple squares
        squares = [create_square(i) for i in range(1, 6)]

        # Verify each square has the correct area
        for i, square in enumerate(squares, 1):
            self.assertEqual(square.side, i)
            self.assertEqual(square.area, i * i)

        # Verify count in database
        self.assertEqual(Square.objects.count(), 5)

    def test_zero_side(self):
        # Test with side=0
        square = create_square(0)
        self.assertEqual(square.side, 0)
        self.assertEqual(square.area, 0)

    def test_large_number(self):
        # Test with a large number to ensure BigIntegerField works
        large_side = 10000
        square = create_square(large_side)
        self.assertEqual(square.side, large_side)
        self.assertEqual(square.area, large_side * large_side)


if __name__ == "__main__":
    unittest.main()
