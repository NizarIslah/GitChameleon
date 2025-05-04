# Add the parent directory to import sys
import os
import socket  # socket is already imported, which is needed for socket.gaierror
import sys
import unittest

import tornado.ioloop
import tornado.netutil
import tornado.testing
import tornado.web
import tornado.websocket

# Ensure the path is set up correctly to find sample_260
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sample_260 import custom_websocket_connect


class TestWebSocketHandler(tornado.websocket.WebSocketHandler):
    """Test WebSocket handler for the server side."""

    def open(self):
        self.write_message("WebSocket connection established")

    def on_message(self, message):
        self.write_message(f"Echo: {message}")

    def on_close(self):
        pass


class TestCustomWebSocketConnect(tornado.testing.AsyncHTTPTestCase):
    """Test case for custom_websocket_connect function."""

    def get_app(self):
        """Create a tornado application with a WebSocket handler."""
        return tornado.web.Application(
            [
                (r"/ws", TestWebSocketHandler),
            ]
        )

    @tornado.testing.gen_test
    async def test_custom_websocket_connect(self):
        """Test that custom_websocket_connect successfully connects to a WebSocket server."""
        port = self.get_http_port()

        resolver = tornado.netutil.DefaultLoopResolver()

        ws_url = f"ws://localhost:{port}/ws"
        ws_conn = await custom_websocket_connect(ws_url, resolver)

        self.assertIsInstance(ws_conn, tornado.websocket.WebSocketClientConnection)

        initial_message = await ws_conn.read_message()
        self.assertEqual(initial_message, "WebSocket connection established")

        await ws_conn.write_message("Hello")
        response = await ws_conn.read_message()
        self.assertEqual(response, "Echo: Hello")

        ws_conn.close()

    @tornado.testing.gen_test
    async def test_connection_error(self):
        """Test that custom_websocket_connect handles connection errors properly."""
        resolver = tornado.netutil.DefaultLoopResolver()

        # Changed expected exception from tornado.httpclient.HTTPError to socket.gaierror
        with self.assertRaises(socket.gaierror):
            await custom_websocket_connect(
                "ws://non-existent-server:12345/ws", resolver
            )


if __name__ == "__main__":
    unittest.main()
