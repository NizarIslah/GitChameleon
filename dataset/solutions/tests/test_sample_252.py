import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_252 import custom_parse_query


class TestCustomParseQuery(unittest.TestCase):
    def test_basic_query_parsing(self):
        """Test basic query string parsing."""
        query_string = "name=John&age=30"
        result = custom_parse_query(query_string)
        self.assertEqual(result, {"name": "John", "age": "30"})

    def test_blank_values(self):
        """Test that blank values are kept (keep_blank=True)."""
        query_string = "name=&age=30&email="
        result = custom_parse_query(query_string)
        self.assertEqual(result, {"name": "", "age": "30", "email": ""})

    def test_special_characters(self):
        """Test parsing query string with special characters."""
        query_string = "message=Hello%20World&url=https%3A%2F%2Fexample.com"
        result = custom_parse_query(query_string)
        self.assertEqual(
            result, {"message": "Hello World", "url": "https://example.com"}
        )


if __name__ == "__main__":
    unittest.main()
