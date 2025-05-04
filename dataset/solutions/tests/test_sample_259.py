import concurrent.futures
import os
import socket

# Add the parent directory to import sys
import sys
import unittest

import tornado.httpclient
import tornado.httpserver
import tornado.ioloop
import tornado.wsgi

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_259 import custom_wsgi_container, find_free_port, simple_wsgi_app


class TestSample259(unittest.TestCase):
    def test_simple_wsgi_app(self):
        """Test that simple_wsgi_app returns 'Hello World'"""
        # Mock the start_response function
        start_response_calls = []

        def mock_start_response(status, headers):
            start_response_calls.append((status, headers))

        # Call the WSGI app
        environ = {}  # Empty environment for this simple test
        result = simple_wsgi_app(environ, mock_start_response)

        # Check the response
        self.assertEqual(result, [b"Hello World"])
        self.assertEqual(len(start_response_calls), 1)
        self.assertEqual(start_response_calls[0][0], "200 OK")
        self.assertEqual(start_response_calls[0][1], [("Content-Type", "text/plain")])

    def test_find_free_port(self):
        """Test that find_free_port returns a valid port number"""
        port = find_free_port()
        self.assertIsInstance(port, int)
        self.assertTrue(1024 <= port <= 65535)  # Common range for non-privileged ports

        # Test that the port is actually free by trying to bind to it
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("localhost", port))
                # If we get here, the port was free
                is_free = True
            except OSError:
                is_free = False

            self.assertTrue(is_free, f"Port {port} was not free as expected")

    def test_custom_wsgi_container(self):
        """Test that custom_wsgi_container returns a WSGIContainer with the correct app and executor"""
        # Create a test executor
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        # Create the container
        container = custom_wsgi_container(simple_wsgi_app, executor)

        # Verify it's the right type
        self.assertIsInstance(container, tornado.wsgi.WSGIContainer)

        # In Tornado 6.3.0, the WSGIContainer stores the app and executor as attributes
        # We can verify they were set correctly
        self.assertEqual(container.wsgi_application, simple_wsgi_app)
        self.assertEqual(container.executor, executor)


if __name__ == "__main__":
    unittest.main()
