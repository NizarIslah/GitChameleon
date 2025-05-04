import json
import os
import sys
import tempfile
import unittest

from flask import Flask

# Import the function to test

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_147 import app, load_config


class TestSample147(unittest.TestCase):
    def setUp(self):
        # Create a test Flask app
        self.app = app
        # Create a test client
        self.client = self.app.test_client()
        # Create a temporary config file for testing
        self.test_config = {
            "DEBUG": False,
            "SECRET_KEY": "test_secret",
            "TESTING": True,
        }
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".json"
        )
        json.dump(self.test_config, self.temp_file)
        self.temp_file.flush()
        self.config_file = self.temp_file.name

    def tearDown(self):
        # Clean up the temporary file
        self.temp_file.close()
        os.unlink(self.config_file)

    def test_load_config(self):
        # Test that the config is loaded correctly
        load_config(self.config_file)

        # Check that all config values were loaded
        for key, value in self.test_config.items():
            self.assertEqual(self.app.config[key], value)

        # Verify specific values
        self.assertEqual(self.app.config["DEBUG"], False)
        self.assertEqual(self.app.config["SECRET_KEY"], "test_secret")
        self.assertEqual(self.app.config["TESTING"], True)


if __name__ == "__main__":
    unittest.main()
