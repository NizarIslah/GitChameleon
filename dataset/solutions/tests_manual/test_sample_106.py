# Add the parent directory to import sys
import os
import sys
import unittest
from django.db import connection
from django.db.utils import OperationalError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_106 import Square, create_square, display_side_and_area


with connection.schema_editor() as schema_editor:
    schema_editor.create_model(Square)
square = create_square(side=5)
correct_result = (5, 25)
assert display_side_and_area(square) == correct_result
