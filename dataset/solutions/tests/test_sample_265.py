# Add the parent directory to import sys
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tornado.ioloop
from sample_265 import custom_get_ioloop


class TestSample265(unittest.TestCase):
    def test_custom_get_ioloop(self):
        """Test that custom_get_ioloop returns the current IOLoop instance."""
        # Get the current IOLoop directly
        expected_ioloop = tornado.ioloop.IOLoop.current()

        # Get the IOLoop using our custom function
        result_ioloop = custom_get_ioloop()

        # Verify they are the same instance
        self.assertIs(
            result_ioloop,
            expected_ioloop,
            "custom_get_ioloop should return the current IOLoop instance",
        )

        # Verify the type is correct
        self.assertIsInstance(
            result_ioloop,
            tornado.ioloop.IOLoop,
            "custom_get_ioloop should return an IOLoop instance",
        )


if __name__ == "__main__":
    unittest.main()
