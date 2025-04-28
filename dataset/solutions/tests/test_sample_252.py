import unittest
import sys
import os

# Add the parent directory to import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sample_252 import custom_parse_query


class TestCustomParseQuery(unittest.TestCase):
    
    def test_basic_query_parsing(self):
        """Test basic query string parsing."""
        query_string = "name=John&age=30"
        result = custom_parse_query(query_string)
        self.assertEqual(result, {"name": "John", "age": "30"})
    
    def test_empty_query_string(self):
        """Test parsing an empty query string."""
        query_string = ""
        result = custom_parse_query(query_string)
        self.assertEqual(result, {})
    
    def test_blank_values(self):
        """Test that blank values are kept (keep_blank=True)."""
        query_string = "name=&age=30&email="
        result = custom_parse_query(query_string)
        self.assertEqual(result, {"name": "", "age": "30", "email": ""})
    
    def test_repeated_keys(self):
        """Test handling of repeated keys (csv=False)."""
        query_string = "tag=python&tag=falcon&tag=testing"
        result = custom_parse_query(query_string)
        # With csv=False, the last value should be used
        self.assertEqual(result, {"tag": "testing"})
    
    def test_special_characters(self):
        """Test parsing query string with special characters."""
        query_string = "message=Hello%20World&url=https%3A%2F%2Fexample.com"
        result = custom_parse_query(query_string)
        self.assertEqual(result, {"message": "Hello World", "url": "https://example.com"})
    
    def test_complex_query(self):
        """Test a more complex query string."""
        query_string = "user=admin&role=&filters=active&filters=verified&page=1&limit=10"
        result = custom_parse_query(query_string)
        # With csv=False, the last value for 'filters' should be used
        self.assertEqual(result, {
            "user": "admin",
            "role": "",
            "filters": "verified",
            "page": "1",
            "limit": "10"
        })


if __name__ == "__main__":
    unittest.main()