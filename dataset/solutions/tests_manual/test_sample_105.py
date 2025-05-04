# Add the parent directory to import sys
import os
import sys
import unittest

from uuid import uuid4
from django.apps import AppConfig


class MyAppConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    label = "myapp_" + str(uuid4())


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the module to test
from sample_105 import Square, create_square, display_side_and_area


class TestSquareModel(unittest.TestCase):
    table_exists = False

    def setUp(self):
        # Create the necessary tables in the in-memory database
        from django.db import connection


with connection.schema_editor() as schema_editor:
    schema_editor.create_model(Square)
square = create_square(side=5)
correct_result = (5, 25)
assert display_side_and_area(square) == correct_result
